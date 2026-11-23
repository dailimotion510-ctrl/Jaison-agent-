import os
import re
import json
import time
import signal
import asyncio
import logging
import subprocess

import yt_dlp
from dotenv import load_dotenv
import static_ffmpeg
from telethon import TelegramClient, events
from telethon.sessions import StringSession

load_dotenv()
static_ffmpeg.add_paths()  # deja ffmpeg y ffprobe disponibles en el PATH

# --- CONFIGURACIÓN ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

API_ID = int(os.environ.get("API_ID", 0))
API_HASH = os.environ.get("API_HASH", "")
STRING_SESSION = os.environ.get("STRING_SESSION", "")
# Opcional: IDs de Telegram autorizados, separados por coma. Vacío = todos.
ADMIN_IDS = {int(x) for x in os.environ.get("ADMIN_IDS", "").split(",") if x.strip().isdigit()}

PART_SIZE_MB = 1900  # Tamaño de cada parte en MB (seguro para 2GB)
DOWNLOAD_DIR = 'downloads'
CONFIG_FILE = 'config.json'

client = TelegramClient(StringSession(STRING_SESSION), API_ID, API_HASH)

GRABACIONES = {}  # chat_id -> {'proc': proceso ffmpeg, 'parar': bool}

RE_CANAL = re.compile(r'https?://(?:www\.)?twitch\.tv/([A-Za-z0-9_]+)/?(?:\?.*)?$')
RESERVADOS = {'videos', 'directory', 'downloads', 'settings', 'jobs', 'p'}


# --- CONFIG PERSISTENTE (canal por chat) ---
def cargar_config():
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def guardar_config(cfg):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(cfg, f)
    except Exception as e:
        logger.error(f"No se pudo guardar la config: {e}")


CONFIG = cargar_config()


def extraer_canal(texto):
    """Acepta 'nombre', '@nombre' o 'https://twitch.tv/nombre'."""
    texto = texto.strip()
    m = RE_CANAL.match(texto)
    nombre = (m.group(1) if m else texto.lstrip('@')).lower()
    if re.fullmatch(r'[a-z0-9_]{3,25}', nombre) and nombre not in RESERVADOS:
        return nombre
    return None


def cmd(pattern):
    return events.NewMessage(pattern=pattern, incoming=True, func=lambda e: e.is_private)


def permitido(event):
    return not ADMIN_IDS or event.sender_id in ADMIN_IDS


# --- UTILIDADES DE PROGRESO ---
def fmt_mb(b):
    return f"{b / 1024 / 1024:.0f} MB"


def fmt_tiempo(s):
    h, r = divmod(int(s), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def barra(actual, total, largo=10):
    pct = min(actual * 100 / total, 100) if total else 0
    llenos = int(pct // (100 / largo))
    return f"[{'█' * llenos}{'░' * (largo - llenos)}] {pct:.0f}%"


async def editar(msg, texto):
    try:
        await msg.edit(texto)
    except Exception:
        pass


def hook_descarga(msg, loop):
    ultimo = [0]

    def hook(d):
        if d.get('status') != 'downloading':
            return
        ahora = time.time()
        if ahora - ultimo[0] < 3:
            return
        ultimo[0] = ahora
        actual = d.get('downloaded_bytes', 0)
        total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
        vel = (d.get('_speed_str') or '').strip()
        texto = f"⏳ Descargando...\n{barra(actual, total)}\n💾 {fmt_mb(actual)}"
        if vel:
            texto += f" • ⚡ {vel}"
        asyncio.run_coroutine_threadsafe(editar(msg, texto), loop)

    return hook


def cb_subida(msg, titulo="📤 Subiendo a Telegram"):
    ultimo = [0]

    async def cb(actual, total):
        ahora = time.time()
        if ahora - ultimo[0] < 3 and actual < total:
            return
        ultimo[0] = ahora
        await editar(msg, f"{titulo}\n{barra(actual, total)}")

    return cb


# --- FUNCIÓN PARA DIVIDIR ---
def split_video(input_file, output_dir, part_size_mb):
    part_size_bytes = part_size_mb * 1024 * 1024
    probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
    try:
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
    except Exception as e:
        logger.error(f"Error obteniendo duración: {e}")
        return None

    file_size = os.path.getsize(input_file)
    num_parts = int(file_size // part_size_bytes) + 1
    part_duration = duration / num_parts

    output_files = []
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    for i in range(num_parts):
        start_time = i * part_duration
        output_file = os.path.join(output_dir, f"{base_name}_parte{i + 1}.mp4")
        cmd_ff = [
            'ffmpeg', '-ss', str(start_time), '-i', input_file,
            '-t', str(part_duration), '-c', 'copy', output_file
        ]
        try:
            subprocess.run(cmd_ff, check=True, capture_output=True)
            output_files.append(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error dividiendo parte {i + 1}: {e}")
            return None
    return output_files


# --- ENVÍO (con división si es muy grande) ---
async def enviar_archivo(event, msg, filename, titulo, etiqueta):
    loop = asyncio.get_running_loop()
    size_mb = os.path.getsize(filename) / (1024 * 1024)

    if size_mb <= PART_SIZE_MB:
        await editar(msg, "📤 Subiendo a Telegram...")
        await client.send_file(
            event.chat_id, filename,
            caption=f"✅ **{titulo}**\n📺 {etiqueta}",
            supports_streaming=True,
            progress_callback=cb_subida(msg)
        )
        os.remove(filename)
        return True

    await editar(msg, f"📦 El video pesa {size_mb:.1f} MB. Dividiendo en partes...")
    partes = await loop.run_in_executor(None, split_video, filename, DOWNLOAD_DIR, PART_SIZE_MB)
    if not partes:
        await editar(msg, "❌ Error al dividir el video. Revisa que ffmpeg funcione.")
        os.remove(filename)
        return False

    for idx, parte in enumerate(partes, 1):
        await client.send_file(
            event.chat_id, parte,
            caption=f"✅ **{titulo}** - Parte {idx}/{len(partes)}\n📺 {etiqueta}",
            supports_streaming=True,
            progress_callback=cb_subida(msg, f"📤 Subiendo parte {idx}/{len(partes)}")
        )
        os.remove(parte)
    os.remove(filename)
    return True


# --- COMANDOS ---
@client.on(cmd(r'^/start'))
async def start(event):
    if not permitido(event):
        return
    await event.reply(
        "¡Hola! Soy tu bot descargador de Twitch. 🎮\n\n"
        "📌 **Cómo usarme:**\n"
        "• Envíame un enlace de un **VOD** o **Clip** y lo descargo en 480p.\n"
        "• Elige un canal con `/canal nombre` y usa `/directo` para grabar su directo.\n\n"
        "Si el video pesa mucho, lo divido en partes de 2GB.\n"
        "Usa /ayuda para ver todos los comandos."
    )


@client.on(cmd(r'^/ayuda'))
async def ayuda(event):
    if not permitido(event):
        return
    await event.reply(
        "🛠 **Comandos:**\n\n"
        "`/canal nombre` - Define el canal de Twitch\n"
        "`/canal` - Muestra el canal actual\n"
        "`/directo` - Graba el directo del canal hasta que termine\n"
        "`/directo 30` - Graba 30 minutos (con barra de %)\n"
        "`/parar` - Detiene la grabación y envía lo grabado\n\n"
        "También puedes enviar:\n"
        "- `https://www.twitch.tv/videos/123456789` (VOD)\n"
        "- `https://clips.twitch.tv/NombreDelClip` (Clip)\n"
        "- `https://www.twitch.tv/nombre` (graba el directo de ese canal)"
    )


@client.on(cmd(r'^/canal(?:\s+(\S+))?\s*$'))
async def canal(event):
    if not permitido(event):
        return
    arg = event.pattern_match.group(1)
    clave = str(event.chat_id)

    if not arg:
        actual = CONFIG.get(clave)
        if actual:
            await event.reply(f"📺 Canal actual: **{actual}**\nCámbialo con `/canal nombre`.")
        else:
            await event.reply("No hay canal definido. Usa `/canal nombre`.")
        return

    nombre = extraer_canal(arg)
    if not nombre:
        await event.reply("❌ Nombre de canal no válido. Ejemplo: `/canal ibai`")
        return

    CONFIG[clave] = nombre
    guardar_config(CONFIG)
    await event.reply(f"✅ Canal guardado: **{nombre}**\nUsa `/directo` para grabar su directo.")


@client.on(cmd(r'^/directo(?:\s+(\d+))?\s*$'))
async def directo(event):
    if not permitido(event):
        return
    minutos = int(event.pattern_match.group(1) or 0) or None
    nombre = CONFIG.get(str(event.chat_id))
    if not nombre:
        await event.reply("Primero define un canal con `/canal nombre`.")
        return
    await grabar_directo(event, nombre, minutos)


@client.on(cmd(r'^/parar'))
async def parar(event):
    if not permitido(event):
        return
    rec = GRABACIONES.get(event.chat_id)
    if not rec:
        await event.reply("No hay ninguna grabación en curso.")
        return
    rec['parar'] = True
    proc = rec.get('proc')
    if proc:
        try:
            proc.send_signal(signal.SIGINT)  # ffmpeg cierra el archivo bien con SIGINT
        except ProcessLookupError:
            pass
    await event.reply("🛑 Deteniendo la grabación...")


# --- GRABAR DIRECTO ---
async def grabar_directo(event, canal_nombre, minutos=None):
    chat_id = event.chat_id
    if chat_id in GRABACIONES:
        await event.reply("⚠️ Ya hay una grabación en curso. Usa /parar para detenerla.")
        return

    GRABACIONES[chat_id] = {'proc': None, 'parar': False}
    msg = await event.reply(f"🔎 Buscando el directo de **{canal_nombre}**...")
    loop = asyncio.get_running_loop()
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    try:
        def _info():
            opts = {'format': 'best[height<=480]/best', 'quiet': True,
                    'no_warnings': True, 'noplaylist': True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(f"https://www.twitch.tv/{canal_nombre}", download=False)

        try:
            info = await loop.run_in_executor(None, _info)
            stream_url = info.get('url')
            if not stream_url:
                raise ValueError("Sin URL de stream")
        except Exception as e:
            logger.error(f"Error abriendo directo: {e}")
            await editar(msg, f"❌ **{canal_nombre}** no está en directo o no se pudo abrir.")
            return

        salida = os.path.join(DOWNLOAD_DIR, f"{canal_nombre}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        cmd_ff = ['ffmpeg', '-y', '-nostdin', '-loglevel', 'error',
                  '-i', stream_url, '-c', 'copy', '-bsf:a', 'aac_adtstoasc',
                  '-progress', 'pipe:1', '-nostats']
        if minutos:
            cmd_ff += ['-t', str(minutos * 60)]
        cmd_ff.append(salida)

        proc = await asyncio.create_subprocess_exec(
            *cmd_ff, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        GRABACIONES[chat_id]['proc'] = proc
        if GRABACIONES[chat_id]['parar']:  # /parar llegó antes de que arrancara
            proc.send_signal(signal.SIGINT)

        seg = 0
        ultimo = 0
        while True:
            linea = await proc.stdout.readline()
            if not linea:
                break
            linea = linea.decode(errors='ignore').strip()
            if linea.startswith(('out_time_us=', 'out_time_ms=')):
                try:
                    seg = int(linea.split('=')[1]) / 1_000_000
                except ValueError:
                    pass
            elif linea.startswith('progress='):
                ahora = time.time()
                if ahora - ultimo >= 3:
                    ultimo = ahora
                    tam = os.path.getsize(salida) if os.path.exists(salida) else 0
                    if minutos:
                        texto = (f"🔴 Grabando **{canal_nombre}**\n{barra(seg, minutos * 60)}\n"
                                 f"⏱ {fmt_tiempo(seg)} / {fmt_tiempo(minutos * 60)} • 💾 {fmt_mb(tam)}\n"
                                 "Usa /parar para detener y enviar.")
                    else:
                        texto = (f"🔴 Grabando **{canal_nombre}**\n"
                                 f"⏱ {fmt_tiempo(seg)} • 💾 {fmt_mb(tam)}\n"
                                 "Usa /parar para detener y enviar.")
                    await editar(msg, texto)

        err = (await proc.stderr.read()).decode(errors='ignore')
        await proc.wait()

        if not os.path.exists(salida) or os.path.getsize(salida) == 0:
            await editar(msg, f"❌ No se pudo grabar el directo.\n\n{err[-150:]}")
            return

        await editar(msg, "✅ Grabación terminada. Preparando envío...")
        ok = await enviar_archivo(event, msg, salida, f"Directo de {canal_nombre}", "🔴 Directo 480p")
        if ok:
            await editar(msg, "✅ Enviado.")

    except Exception as e:
        logger.error(f"Error grabando directo: {e}")
        await editar(msg, f"❌ Error al grabar el directo.\n\n*Detalle:* {str(e)[:150]}")
    finally:
        GRABACIONES.pop(chat_id, None)


# --- VOD / CLIP POR ENLACE ---
@client.on(events.NewMessage(
    incoming=True,
    func=lambda e: e.is_private and e.text and 'twitch.tv' in e.text and not e.text.startswith('/')
))
async def descargar_twitch(event):
    if not permitido(event):
        return
    url = event.text.strip()

    # Si es el enlace de un canal (sin /videos ni clip), graba su directo
    m = RE_CANAL.match(url)
    if m and m.group(1).lower() not in RESERVADOS:
        await grabar_directo(event, m.group(1).lower())
        return

    msg = await event.reply("⏳ Preparando descarga en 480p...")
    loop = asyncio.get_running_loop()
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    ydl_opts = {
        'format': 'best[height<=480][ext=mp4]/best[height<=480]/best',
        'outtmpl': f'{DOWNLOAD_DIR}/%(title)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'progress_hooks': [hook_descarga(msg, loop)],
    }

    try:
        def _descargar():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return info, ydl.prepare_filename(info)

        info, filename = await loop.run_in_executor(None, _descargar)
        titulo = info.get('title', 'Video de Twitch')

        ok = await enviar_archivo(event, msg, filename, titulo, "480p")
        if ok:
            await editar(msg, "✅ Enviado.")

    except Exception as e:
        logger.error(f"Error: {e}")
        await editar(msg, f"❌ Error al procesar el video.\n\n*Detalle:* {str(e)[:150]}...")


# --- INICIO ---
async def main():
    if not all([API_ID, API_HASH, STRING_SESSION]):
        print("ERROR: Faltan variables de entorno (API_ID, API_HASH, STRING_SESSION)")
        return

    await client.start()
    print("Bot iniciado con éxito (Userbot MTProto)...")
    await client.run_until_disconnected()


if __name__ == '__main__':
    asyncio.run(main())
