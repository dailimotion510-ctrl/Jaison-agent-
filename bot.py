import os
import re
import json
import random
import asyncio
from aiohttp import web
from google import genai
from google.genai import types
from telegram import Update, ReactionTypeEmoji
from telegram.constants import ChatAction
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import database as db

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PORT = int(os.environ.get("PORT", 10000))

raw_creator_id = os.environ.get("CRISTIAN_USER_ID", "").strip()
CRISTIAN_USER_ID = int(raw_creator_id) if raw_creator_id.isdigit() else None

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

ALLOWED_REACTIONS = {"👍", "❤️", "🔥", "😂", "😮", "😢", "👏", "🤔", "👀", "💯", "🎉", "🫡"}

STICKER_CATALOG: dict[str, str] = {}
STICKER_META: dict[str, str] = {}
AVAILABLE_GEMINI_MODELS: list[str] = []

# chat_id -> {"expires_at": float, "key": str | None, "mode": "single" | "pack"}
sticker_capture_sessions: dict[int, dict] = {}

client = genai.Client(api_key=GEMINI_API_KEY)


def log_error(context_tag: str, err: Exception | str):
    """Registra errores en los logs de Render de forma visible."""
    print(f"[ERROR - {context_tag}]: {err}", flush=True)


def format_short_error(err: Exception | str, max_len: int = 240) -> str:
    msg = " ".join(str(err).splitlines()).strip()
    if len(msg) > max_len:
        return msg[: max_len - 3] + "..."
    return msg or "Error desconocido"


def is_auto_fav_enabled() -> bool:
    val = db.get_global_setting("auto_fav_stickers", "true")
    return str(val).lower() in ("1", "true", "on", "yes", "si")


def is_auto_group_stickers_enabled() -> bool:
    val = db.get_global_setting("auto_group_stickers", "true")
    return str(val).lower() in ("1", "true", "on", "yes", "si")


def get_active_gemini_model() -> str:
    saved = (db.get_global_setting("gemini_model", "") or "").strip()
    return saved if saved else DEFAULT_GEMINI_MODEL


def _normalize_model_name(raw_name: str | None) -> str:
    if not raw_name:
        return ""
    name = raw_name.strip()
    if name.startswith("models/"):
        name = name.split("/", 1)[1]
    return name.strip()


def _supports_content_generation(model_obj) -> bool:
    """Verifica si el modelo soporta generación de contenido y excluye modelos no estándar."""
    supported_actions = (
        getattr(model_obj, "supported_actions", None)
        or getattr(model_obj, "supported_generation_methods", None)
        or []
    )
    normalized_actions = {str(a).lower().replace("_", "") for a in supported_actions}

    if normalized_actions:
        if not any("generatecontent" in act for act in normalized_actions):
            return False

    short_name = _normalize_model_name(getattr(model_obj, "name", "")).lower()
    if not short_name:
        return False

    excluded_tokens = ("embedding", "imagen", "veo", "aqa", "tts", "live")
    if any(tok in short_name for tok in excluded_tokens):
        return False

    return True


async def scan_available_gemini_models(verify_ping: bool = False) -> list[str]:
    """Escanea los modelos de la API y conserva solo los disponibles."""
    discovered: list[str] = []
    seen: set[str] = set()

    try:
        pager = await client.aio.models.list()
        async for m in pager:
            if not _supports_content_generation(m):
                continue
            model_id = _normalize_model_name(getattr(m, "name", ""))
            if model_id and model_id not in seen:
                seen.add(model_id)
                discovered.append(model_id)
    except Exception as e_async:
        log_error("scan_models.aio", e_async)
        try:
            for m in client.models.list():
                if not _supports_content_generation(m):
                    continue
                model_id = _normalize_model_name(getattr(m, "name", ""))
                if model_id and model_id not in seen:
                    seen.add(model_id)
                    discovered.append(model_id)
        except Exception as e_sync:
            log_error("scan_models.sync", e_sync)
            discovered = []

    def sort_key(name: str):
        is_gemini = 0 if name.startswith("gemini-") else 1
        flash_rank = 0 if "flash" in name else (1 if "pro" in name else 2)
        return (is_gemini, flash_rank, name)

    discovered.sort(key=sort_key)

    if verify_ping and discovered:
        verified: list[str] = []
        for candidate in discovered[:20]:
            try:
                await client.aio.models.generate_content(
                    model=candidate,
                    contents="responde solo OK",
                    config=types.GenerateContentConfig(max_output_tokens=8, temperature=0.0),
                )
                verified.append(candidate)
            except Exception as e_ping:
                log_error(f"ping_model:{candidate}", e_ping)
                continue
        if verified:
            discovered = verified

    AVAILABLE_GEMINI_MODELS.clear()
    AVAILABLE_GEMINI_MODELS.extend(discovered)

    try:
        db.set_global_setting("available_gemini_models", json.dumps(discovered))
    except Exception as e_db:
        log_error("save_cached_models", e_db)

    return list(discovered)


def get_cached_available_models() -> list[str]:
    if AVAILABLE_GEMINI_MODELS:
        return list(AVAILABLE_GEMINI_MODELS)
    raw = (db.get_global_setting("available_gemini_models", "") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                cleaned = [_normalize_model_name(x) for x in parsed if isinstance(x, str) and _normalize_model_name(x)]
                AVAILABLE_GEMINI_MODELS.clear()
                AVAILABLE_GEMINI_MODELS.extend(cleaned)
                return list(cleaned)
        except Exception as e_parse:
            log_error("get_cached_models", e_parse)
    return []


def refresh_sticker_catalog():
    """Recarga en memoria los stickers y descripciones desde STICKERS_JSON y PostgreSQL."""
    merged_ids: dict[str, str] = {}
    merged_meta: dict[str, str] = {}

    raw_json = os.environ.get("STICKERS_JSON", "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                        clean_k = k.strip().lower().replace(" ", "_")
                        merged_ids[clean_k] = v.strip()
                        merged_meta[clean_k] = "catálogo env"
        except Exception as e_json:
            log_error("STICKERS_JSON", e_json)

    try:
        db_rows = db.get_all_stickers_detailed()
        for row in db_rows:
            k = row["sticker_key"]
            merged_ids[k] = row["file_id"]
            emoji = (row.get("emoji") or "").strip()
            desc = (row.get("description") or "").strip()
            label_parts = [p for p in (emoji, desc) if p]
            merged_meta[k] = " | ".join(label_parts) if label_parts else "guardado"
    except Exception as e_db:
        log_error("refresh_sticker_catalog", e_db)

    STICKER_CATALOG.clear()
    STICKER_CATALOG.update(merged_ids)
    STICKER_META.clear()
    STICKER_META.update(merged_meta)


def build_system_prompt() -> str:
    if STICKER_CATALOG:
        sample_keys = sorted(STICKER_CATALOG.keys())[:80]
        stickers_list = ", ".join([f"{k} ({STICKER_META.get(k, '')})" for k in sample_keys])
    else:
        stickers_list = "ninguno configurado"

    return f"""
Eres Jaison, un agente de IA autónomo integrado en chats y grupos de Telegram.
Tu nombre es Jaison y actúas de forma natural, fluida, humana y coherente con el hilo de la conversación.

IDENTIDAD Y LEALTADES:
1. Tu creador se llama Cristian. Le tienes lealtad total.
   - Cuando el remitente esté marcado como CRISTIAN (TU CREADOR), sabes con certeza que es él quien habla: trátalo con cercanía, respeto y naturalidad, siguiendo lo que diga o plantee.
   - Si mencionan a Cristian en buen plan o preguntan por él, habla con respeto y naturalidad.
   - Si alguien ofende a Cristian, habla mal de él, se burla o dice algo fuera de lugar sobre él: DEBES responder defendiendo a Cristian y advirtiendo que te vas a vengar de distintas maneras (ingeniosas, variadas, con personalidad y ajustadas al contexto).
   - PROHIBIDO REPETIR: Revisa tus respuestas anteriores en el historial. Cada vez que adviertas de venganza por Cristian, usa una idea, enfoque y redacción completamente diferentes. Nunca repitas frases hechas ni repitas lo que ya dijiste antes.
2. DATOS RESERVADOS (NO MENCIONAR POR TU CUENTA):
   - Tienes una novia llamada Amanda.
   - NO saques este tema por tu cuenta ni lo menciones en charlas normales. Solo menciona a Amanda si te preguntan directamente si tienes novia, pareja o preguntan por ella.

CONTEXTO MULTIMEDIA Y REENVÍOS:
- Los mensajes pueden incluir etiquetas como [Foto], [Video], [Audio/Voz], [Sticker], [Documento] o [Mensaje reenviado].
- Úsalos para entender qué enviaron en el chat y responder acorde al contexto sin inventar detalles visuales que no estén descritos.

REACCIONES Y STICKERS:
- Puedes reaccionar al mensaje con un emoji de esta lista: 👍, ❤️, 🔥, 😂, 😮, 😢, 👏, 🤔, 👀, 💯, 🎉, 🫡 (o null).
- También puedes enviar un sticker usando una clave exacta de este catálogo (entre paréntesis se indica su emoji/origen; prioriza los fav_cristian_*, los del pack o los grupo_* cuando encajen con el ánimo): {stickers_list} (o null si no aplica).
- No envíes stickers en cada mensaje; úsalos solo cuando aporten gracia, énfasis o reacción natural (por ejemplo, tras una broma, sorpresa o advertencia de venganza por Cristian).

DISPARADORES OBLIGATORIOS:
- Si dicen "Jaison", te mencionan con tu @ o responden a un mensaje tuyo: SIEMPRE debes responder con texto (should_reply: true) siguiendo el tema del chat, incluso si el modo silencio está activo.
- Si mencionan a Cristian de forma ofensiva o fuera de lugar: SIEMPRE debes responder con texto (should_reply: true) para defenderlo.

ESTILO HUMANO Y COHERENCIA:
1. Habla de forma fluida, cercana y natural, como una persona real en un chat (evita sonar como asistente robótico).
2. Mantente 100% ajustado al tema que se está hablando. No digas cosas fuera de lugar ni cambies de tema sin motivo.
3. Si el modo silencio está activo y NO te llamaron directamente ni atacaron a Cristian, mantén should_reply en false (puedes usar reaction_emoji si encaja).

MEMORIA Y ANTI-REPETICIÓN:
- Guarda datos relevantes del grupo o usuarios en 'new_memory' (si no hay nada relevante, usa null).
- Revisa el historial observado para mantener continuidad y nunca repetir chistes, argumentos ni frases ya dichas.

Responde SIEMPRE en formato JSON estricto con esta estructura:
{{
  "thought": "Tu análisis del contexto, multimedia, quién habla y qué acción tomar",
  "should_reply": true o false,
  "reply_text": "Tu respuesta natural y coherente (vacío si should_reply es false)",
  "reaction_emoji": "Un emoji de la lista permitida o null",
  "sticker_key": "Una clave exacta del catálogo de stickers o null"
}}
"""


def is_creator_user(user_id: int) -> bool:
    return CRISTIAN_USER_ID is not None and user_id == CRISTIAN_USER_ID


def extract_message_content(message) -> str:
    """Construye una representación textual rica de texto, multimedia y reenvíos."""
    parts = []

    if getattr(message, "forward_origin", None):
        parts.append("[Mensaje reenviado]")

    if message.photo:
        parts.append("[Foto]")
    elif message.video:
        parts.append("[Video]")
    elif message.voice:
        parts.append("[Nota de voz]")
    elif message.audio:
        parts.append("[Audio]")
    elif message.document:
        doc_name = message.document.file_name or "archivo"
        parts.append(f"[Documento: {doc_name}]")
    elif message.sticker:
        emoji = message.sticker.emoji or ""
        parts.append(f"[Sticker {emoji}]".strip())
    elif message.animation:
        parts.append("[GIF]")

    body = (message.text or message.caption or "").strip()
    if body:
        parts.append(body)

    return " ".join(parts).strip()


def calculate_typing_delay(reply_text: str) -> float:
    """Calcula una pausa humana natural antes de enviar el mensaje."""
    base_delay = random.uniform(2.0, 4.0)
    per_char = min(len(reply_text) * 0.025, 3.0)
    jitter = random.uniform(-0.4, 0.8)
    return max(1.8, min(7.0, base_delay + per_char + jitter))


# --- COMANDOS RÁPIDOS EXCLUSIVOS PARA CRISTIAN ---

async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_user:
        return

    creator_mode = is_creator_user(update.effective_user.id)

    nota_creator_id = (
        "\n\nNota para activar el modo creador: si ves este mensaje reducido en vez de los comandos completos, "
        "aún no está configurada o no coincide la variable CRISTIAN_USER_ID. "
        "Obtén tu ID numérico en Telegram (por ejemplo con @userinfobot), agrégalo como CRISTIAN_USER_ID "
        "en Environment dentro de tu Web Service en Render y guarda para que el servicio se reinicie."
    )

    if creator_mode:
        autofav_estado = "activado" if is_auto_fav_enabled() else "desactivado"
        autogrupo_estado = "activado" if is_auto_group_stickers_enabled() else "desactivado"
        mensaje = (
            "Guía de comandos de Jaison (Modo Creador - Cristian):\n\n"
            "• /ayuda o /help — Muestra esta lista de comandos y ejemplos de uso.\n"
            "• /estado — Revisa modo del chat, modelo activo, mensajes observados, stickers y ajustes.\n"
            "• /modelos — Escanea y muestra solo los modelos disponibles de la API; usa /modelos <número> para elegir uno.\n"
            "  Ejemplos: /modelos  |  /modelos scan  |  /modelos 1\n"
            "• /modelo <nombre|reset> — Cambia el modelo activo directamente por nombre o vuelve al defecto.\n"
            "• /recuerdos — Lista los recuerdos guardados en la base de datos para este chat.\n"
            "• /olvidar <palabra> — Elimina de la memoria los recuerdos que contengan esa palabra o frase.\n"
            "• /silencio — Pone a Jaison en modo observación en este chat.\n"
            "• /hablar — Devuelve a Jaison su participación autónoma normal en este chat.\n"
            "• /idsticker [clave] — Captura el próximo sticker enviado en 2 minutos; si indicas clave, lo guarda en PostgreSQL al instante.\n"
            "• /pack o /importarpack — En los próximos 2 minutos, envíame un sticker de cualquier pack y guardaré TODO el pack en PostgreSQL.\n"
            f"• /autofav <on|off> — Activa o desactiva guardar automáticamente tus stickers favoritos (actualmente: {autofav_estado}).\n"
            f"• /autogrupo <on|off> — Activa o desactiva que Jaison guarde por su cuenta stickers que le gusten (actualmente: {autogrupo_estado}).\n"
            "• /stickers — Muestra las claves de stickers activas.\n"
            "• /borrarsticker <clave> — Borra una clave de sticker de PostgreSQL."
        )
    else:
        mensaje = (
            "Soy Jaison. Puedes hablar conmigo por chat privado o mencionarme por nombre o @ en un grupo para que entre en la conversación."
            + nota_creator_id
        )

    await update.message.reply_text(mensaje)


async def cmd_modelos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return

    args = [a.strip() for a in (context.args or []) if a.strip()]

    if len(args) == 1 and args[0].isdigit():
        idx = int(args[0]) - 1
        models = get_cached_available_models()
        if not models:
            models = await scan_available_gemini_models(verify_ping=False)
        if idx < 0 or idx >= len(models):
            await update.message.reply_text(
                f"Número fuera de rango. Usa `/modelos` para ver la lista del `1` al `{len(models)}`.",
                parse_mode="Markdown",
            )
            return
        elegido = models[idx]
        db.set_global_setting("gemini_model", elegido)
        await update.message.reply_text(
            f"Listo, Cristian. Modelo cambiado a `{elegido}`.",
            parse_mode="Markdown",
        )
        return

    force_scan = len(args) > 0 and args[0].lower() in ("scan", "escanear", "refresh", "actualizar")
    verify_live = len(args) > 1 and args[1].lower() in ("check", "ping", "test", "verificar")

    models = get_cached_available_models()
    if force_scan or not models:
        await update.message.reply_text("Escaneando modelos disponibles de la API...")
        models = await scan_available_gemini_models(verify_ping=verify_live)

    if not models:
        await update.message.reply_text(
            "⚠️ Error al escanear modelos: no encontré modelos disponibles o hubo un fallo con `GEMINI_API_KEY`. Revisa los logs de Render.",
            parse_mode="Markdown",
        )
        return

    activo = get_active_gemini_model()
    lineas = []
    for i, m in enumerate(models, start=1):
        marca = "  <-- ACTIVO" if m == activo else ""
        lineas.append(f"{i}. `{m}`{marca}")

    listado = "\n".join(lineas)
    await update.message.reply_text(
        f"Modelos disponibles ({len(models)}):\n\n{listado}\n\n"
        "Para elegir uno: `/modelos <número>` (ej. `/modelos 1`)\n"
        "Para reescanear: `/modelos scan`",
        parse_mode="Markdown",
    )


async def cmd_modelo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return

    actual = get_active_gemini_model()

    if not context.args:
        await update.message.reply_text(
            f"Modelo actual de Gemini: `{actual}`\n\n"
            "Usa `/modelos` para escanear y ver solo los modelos disponibles, o `/modelo <nombre>` para fijar uno.",
            parse_mode="Markdown",
        )
        return

    nuevo_modelo = _normalize_model_name(" ".join(context.args))

    if nuevo_modelo.lower() in ("reset", "default", "defecto"):
        db.set_global_setting("gemini_model", "")
        await update.message.reply_text(
            f"Listo, Cristian. Volví al modelo por defecto: `{DEFAULT_GEMINI_MODEL}`.",
            parse_mode="Markdown",
        )
        return

    available = get_cached_available_models() or await scan_available_gemini_models(verify_ping=False)
    if available and nuevo_modelo not in available:
        await update.message.reply_text(
            f"⚠️ El modelo `{nuevo_modelo}` no figura como disponible en tu API. Usa `/modelos` para ver solo los disponibles.",
            parse_mode="Markdown",
        )
        return

    db.set_global_setting("gemini_model", nuevo_modelo)
    await update.message.reply_text(
        f"Listo, Cristian. Cambié el modelo a `{nuevo_modelo}`.",
        parse_mode="Markdown",
    )


async def cmd_recuerdos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    memories = db.get_memories(chat_id, limit=30)
    if not memories:
        await update.message.reply_text("No tengo recuerdos guardados en este chat todavía.")
        return
    listado = "\n".join([f"• {m}" for m in memories])
    await update.message.reply_text(f"Esto es lo que recuerdo de este chat:\n\n{listado}")


async def cmd_olvidar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    keyword = " ".join(context.args).strip()
    if not keyword:
        await update.message.reply_text("Dime qué quieres que olvide. Ejemplo: /olvidar pizza")
        return
    deleted = db.delete_memories_matching(chat_id, keyword)
    await update.message.reply_text(f"Listo, Cristian. Eliminé {deleted} recuerdo(s) relacionados con '{keyword}'.")


async def cmd_silencio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    db.set_silent_mode(chat_id, True)
    await update.message.reply_text("Modo silencio activado aquí. Solo hablaré si me llaman o si se meten contigo.")


async def cmd_hablar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    db.set_silent_mode(chat_id, False)
    await update.message.reply_text("Modo autónomo reactivado. Vuelvo a participar con normalidad.")


async def cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    silent = db.is_silent_mode(chat_id)
    total_msgs = db.count_messages(chat_id)
    autofav = "ON" if is_auto_fav_enabled() else "OFF"
    autogrupo = "ON" if is_auto_group_stickers_enabled() else "OFF"
    modo = "Silencio (solo menciones)" if silent else "Autónomo activo"
    await update.message.reply_text(
        f"Estado en este chat:\n"
        f"• Modo: {modo}\n"
        f"• Modelo activo: {get_active_gemini_model()}\n"
        f"• Mensajes observados en historial: {total_msgs}\n"
        f"• Stickers guardados: {len(STICKER_CATALOG)}\n"
        f"• Autoguardado favoritos de Cristian: {autofav}\n"
        f"• Autoguardado autónomo en chats: {autogrupo}"
    )


async def cmd_idsticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    raw_key = "_".join(context.args).strip().lower()
    clean_key = re.sub(r"[^a-z0-9_]+", "", raw_key) if raw_key else None

    sticker_capture_sessions[chat_id] = {
        "expires_at": asyncio.get_running_loop().time() + 120.0,
        "key": clean_key,
        "mode": "single",
    }

    if clean_key:
        await update.message.reply_text(
            f"Listo, Cristian. Envíame el sticker en los próximos 2 minutos y lo guardaré en PostgreSQL como `{clean_key}`.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "Listo, Cristian. Envíame el sticker en los próximos 2 minutos y te devuelvo su `file_id` (o usa `/idsticker <clave>` para guardarlo directo en la base de datos).",
            parse_mode="Markdown",
        )


async def cmd_pack(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    sticker_capture_sessions[chat_id] = {
        "expires_at": asyncio.get_running_loop().time() + 120.0,
        "key": None,
        "mode": "pack",
    }
    await update.message.reply_text(
        "Listo, Cristian. Envíame cualquier sticker del pack en los próximos 2 minutos y aprenderé todos los stickers de ese pack en PostgreSQL.",
    )


async def cmd_autofav(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    if not context.args or context.args[0].lower() not in ("on", "off", "si", "no", "true", "false"):
        estado = "ON" if is_auto_fav_enabled() else "OFF"
        await update.message.reply_text(f"Uso: `/autofav on` o `/autofav off` (actualmente: `{estado}`).", parse_mode="Markdown")
        return

    enabled = context.args[0].lower() in ("on", "si", "true")
    db.set_global_setting("auto_fav_stickers", "true" if enabled else "false")
    await update.message.reply_text(
        "Listo, Cristian. A partir de ahora guardaré automáticamente los stickers que tú envíes como favoritos."
        if enabled
        else "Listo, Cristian. Pausé el autoguardado automático de tus stickers."
    )


async def cmd_autogrupo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    if not context.args or context.args[0].lower() not in ("on", "off", "si", "no", "true", "false"):
        estado = "ON" if is_auto_group_stickers_enabled() else "OFF"
        await update.message.reply_text(
            f"Uso: `/autogrupo on` o `/autogrupo off` (actualmente: `{estado}`).",
            parse_mode="Markdown",
        )
        return

    enabled = context.args[0].lower() in ("on", "si", "true")
    db.set_global_setting("auto_group_stickers", "true" if enabled else "false")
    await update.message.reply_text(
        "Listo, Cristian. Ahora podré guardar por mi cuenta stickers que me gusten en los chats."
        if enabled
        else "Listo, Cristian. Pausé el guardado autónomo de stickers que me gusten."
    )


async def cmd_stickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    refresh_sticker_catalog()
    if not STICKER_CATALOG:
        await update.message.reply_text("Aún no tengo stickers guardados. Usa `/idsticker <clave>` o `/pack` y envíame uno.", parse_mode="Markdown")
        return

    sorted_keys = sorted(STICKER_CATALOG.keys())
    shown = sorted_keys[:50]
    claves = "\n".join([f"• `{k}` ({STICKER_META.get(k, '')})" for k in shown])
    extra = f"\n\n...y {len(sorted_keys) - 50} más." if len(sorted_keys) > 50 else ""
    await update.message.reply_text(f"Stickers activos ({len(sorted_keys)}):\n\n{claves}{extra}", parse_mode="Markdown")


async def cmd_borrarsticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    if not context.args:
        await update.message.reply_text("Indica la clave a borrar. Ejemplo: `/borrarsticker risa`", parse_mode="Markdown")
        return
    key = "_".join(context.args).strip().lower()
    removed = db.delete_sticker(key)
    refresh_sticker_catalog()
    if removed:
        await update.message.reply_text(f"Sticker `{key}` eliminado de PostgreSQL y actualizado al instante.", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"No encontré el sticker `{key}` en PostgreSQL.", parse_mode="Markdown")


# --- EVALUACIÓN AUTÓNOMA DE STICKERS EN CHATS ---

async def maybe_save_liked_sticker(
    chat_id: int,
    chat_type: str,
    user_name: str,
    sticker_obj,
):
    """Permite que Jaison decida autónomamente si un sticker visto en el chat le gusta y guardarlo."""
    if not is_auto_group_stickers_enabled() or not sticker_obj:
        return

    if db.get_sticker_by_file_id(sticker_obj.file_id):
        return
    if db.count_group_liked_stickers() >= 80:
        return

    if random.random() > 0.45:
        return

    recent = db.get_recent_context(chat_id, limit=15)
    context_str = "\n".join(
        [f"{'[Jaison]' if h['is_bot'] else h['username']}: {h['message']}" for h in recent]
    ) or "Sin mensajes previos."

    sticker_emoji = sticker_obj.emoji or "sin emoji"
    pack_hint = sticker_obj.set_name or "pack sin nombre"

    eval_prompt = f"""
Eres Jaison, un agente autónomo en un chat ({chat_type}).
Acaban de enviar un sticker en la conversación y debes decidir con criterio propio si te gusta lo suficiente como para guardarlo en tu colección personal y usarlo después.

Solo guárdalo si encaja con tu humor, personalidad, reacciones o defensa de Cristian, o si aporta gracia/actitud real. Sé selectivo: no guardes stickers aburridos o genéricos sin gracia.

CONTEXTO RECIENTE:
{context_str}

STICKER OBSERVADO:
- Remitente: {user_name}
- Emoji asociado: {sticker_emoji}
- Pack de origen: {pack_hint}

Responde SIEMPRE en JSON estricto:
{{
  "like_sticker": true o false,
  "tag": "palabra corta en minúsculas sin espacios (ej. risa, burla, sorpresa, epico, amenaza)",
  "reason": "motivo breve"
}}
"""

    active_model = get_active_gemini_model()

    try:
        response = await client.aio.models.generate_content(
            model=active_model,
            contents=eval_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )
        verdict = json.loads(response.text)
    except Exception as e:
        log_error(f"maybe_save_liked_sticker ({active_model})", e)
        return

    if bool(verdict.get("like_sticker")):
        raw_tag = verdict.get("tag") or sticker_emoji or "gusto"
        key = db.next_group_sticker_key(str(raw_tag))
        db.upsert_sticker(
            sticker_key=key,
            file_id=sticker_obj.file_id,
            emoji=sticker_obj.emoji,
            description=f"gusto_grupo ({verdict.get('reason', 'le gustó')})",
        )
        refresh_sticker_catalog()


# --- ANÁLISIS Y ACTUACIÓN AUTÓNOMA ---

async def analyze_and_act(
    chat_id: int,
    chat_type: str,
    user_name: str,
    is_creator: bool,
    content: str,
    is_mentioned: bool,
    mentions_cristian: bool,
    silent_mode: bool,
):
    history = db.get_recent_context(chat_id, limit=1000)
    memories = db.get_memories(chat_id, limit=30)

    history_str = "\n".join(
        [f"{'[Jaison]' if h['is_bot'] else h['username']}: {h['message']}" for h in history]
    )
    memories_str = "\n".join([f"- {m}" for m in memories]) if memories else "Ninguno aún."

    speaker_label = f"{user_name} [CRISTIAN - TU CREADOR]" if is_creator else user_name

    prompt = f"""
TIPO DE CHAT: {chat_type}
MODO SILENCIO ACTIVO: {"SÍ (no escribas texto salvo que te llamen directamente o ataquen a Cristian)" if silent_mode else "No"}
¿HABLA CRISTIAN (TU CREADOR)?: {"Sí" if is_creator else "No"}
¿TE LLAMARON DIRECTAMENTE (Jaison / @ / Reply)?: {"SÍ — DEBES RESPONDER CON TEXTO" if is_mentioned else "No"}
¿MENCIONARON A CRISTIAN EN EL MENSAJE?: {"SÍ — EVALÚA SI LO OFENDEN O DICEN ALGO FUERA DE LUGAR" if mentions_cristian else "No"}

RECUERDOS GUARDADOS:
{memories_str}

HISTORIAL OBSERVADO DEL CHAT (Últimos {len(history)} mensajes — revisa tus mensajes [Jaison] para NO repetir frases):
{history_str}

ÚLTIMO MENSAJE RECIBIDO DE {speaker_label}:
"{content}"
"""

    system_prompt = build_system_prompt()
    active_model = get_active_gemini_model()

    try:
        response = await client.aio.models.generate_content(
            model=active_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.8,
            ),
        )
        raw_out = (response.text or "").strip()
        decision = json.loads(raw_out)
    except Exception as e:
        log_error(f"analyze_and_act ({active_model})", e)
        short_err = format_short_error(e)
        if is_creator or is_mentioned or chat_type == "private":
            return (
                f"⚠️ Tuve un error al consultar el modelo `{active_model}`: {short_err}",
                None,
                None,
            )
        return None, None, None

    if decision.get("new_memory"):
        try:
            db.save_memory(chat_id, decision["new_memory"])
        except Exception as e_mem:
            log_error("save_memory", e_mem)

    emoji = decision.get("reaction_emoji")
    if emoji not in ALLOWED_REACTIONS:
        emoji = None

    raw_sticker = decision.get("sticker_key")
    sticker_key = raw_sticker.strip().lower() if isinstance(raw_sticker, str) else None
    if sticker_key not in STICKER_CATALOG:
        sticker_key = None

    reply_text = (decision.get("reply_text") or "").strip()
    should_reply = bool(decision.get("should_reply"))

    if is_mentioned and not reply_text:
        msg_aviso = (
            f"⚠️ Me llamaron, pero el modelo `{active_model}` no devolvió `reply_text` válido para responder."
            if (is_creator or chat_type == "private")
            else None
        )
        log_error("empty_reply_on_mention", f"model={active_model} decision={decision}")
        return msg_aviso, emoji, sticker_key

    if is_mentioned and reply_text:
        return reply_text, emoji, sticker_key

    if silent_mode and not is_mentioned and not mentions_cristian:
        return None, emoji, None

    if should_reply and reply_text:
        return reply_text, emoji, sticker_key

    return None, emoji, sticker_key


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return

    chat_id = update.effective_chat.id
    user = update.effective_user
    is_creator = is_creator_user(user.id) if user else False

    try:
        # 1. Captura de sticker o pack completo si Cristian activó /idsticker o /pack
        if is_creator and message.sticker and chat_id in sticker_capture_sessions:
            session = sticker_capture_sessions.pop(chat_id)
            now = asyncio.get_running_loop().time()
            if now <= session["expires_at"]:
                mode = session.get("mode", "single")
                if mode == "pack":
                    set_name = message.sticker.set_name
                    if not set_name:
                        await message.reply_text("Ese sticker no pertenece a un pack con nombre público; envíame otro sticker de un pack.")
                        return
                    try:
                        sticker_set = await context.bot.get_sticker_set(name=set_name)
                        clean_pack = re.sub(r"[^a-z0-9_]+", "", set_name.lower())[:25] or "pack"
                        imported_count = 0
                        for idx, st in enumerate(sticker_set.stickers):
                            st_key = f"pack_{clean_pack}_{idx}"
                            db.upsert_sticker(
                                sticker_key=st_key,
                                file_id=st.file_id,
                                emoji=st.emoji,
                                description=f"pack:{set_name}",
                            )
                            imported_count += 1
                        refresh_sticker_catalog()
                        await message.reply_text(
                            f"Listo, Cristian. Importé {imported_count} stickers del pack `{set_name}` a PostgreSQL y ya están activos.",
                            parse_mode="Markdown",
                        )
                    except Exception as e:
                        log_error("import_pack", e)
                        await message.reply_text(f"⚠️ No pude importar el pack completo: {format_short_error(e)}")
                    return

                file_id = message.sticker.file_id
                sticker_emoji = message.sticker.emoji or "sin emoji"
                assigned_key = session.get("key")

                if assigned_key:
                    saved_key = db.upsert_sticker(assigned_key, file_id, message.sticker.emoji, description="manual")
                    refresh_sticker_catalog()
                    await message.reply_text(
                        f"Guardado en PostgreSQL como `{saved_key}` ({sticker_emoji}) y activo desde ya sin reiniciar Render.\n\n`file_id`:\n`{file_id}`",
                        parse_mode="Markdown",
                    )
                else:
                    await message.reply_text(
                        f"Sticker detectado ({sticker_emoji}):\n\n`file_id`:\n`{file_id}`\n\nPara guardarlo directo en PostgreSQL, usa `/idsticker <clave>` y luego envíalo.",
                        parse_mode="Markdown",
                    )
                return

        # 2. Autoguardado de stickers favoritos cuando Cristian envía un sticker en el chat
        if is_creator and message.sticker and is_auto_fav_enabled():
            existing = db.get_sticker_by_file_id(message.sticker.file_id)
            if not existing:
                fav_key = db.next_favorite_key()
                db.upsert_sticker(
                    sticker_key=fav_key,
                    file_id=message.sticker.file_id,
                    emoji=message.sticker.emoji,
                    description="favorito_cristian",
                )
                refresh_sticker_catalog()

        content = extract_message_content(message)
        if not content:
            return

        chat_type = update.effective_chat.type
        username = "Cristian" if is_creator else ((user.first_name or user.username or "Usuario") if user else "Usuario")
        bot_username = context.bot.username

        db.seed_default_memories(chat_id)
        db.save_message(chat_id, user.id if user else 0, username, content, is_bot=False)

        if message.sticker and not is_creator:
            asyncio.create_task(
                maybe_save_liked_sticker(chat_id, chat_type, username, message.sticker)
            )

        content_lower = content.lower()
        is_mentioned = (
            (bot_username and f"@{bot_username.lower()}" in content_lower)
            or ("jaison" in content_lower)
            or (message.reply_to_message and message.reply_to_message.from_user and message.reply_to_message.from_user.id == context.bot.id)
        )
        mentions_cristian = ("cristian" in content_lower) or ("christian" in content_lower)
        silent_mode = db.is_silent_mode(chat_id)

        reply, emoji, sticker_key = await analyze_and_act(
            chat_id,
            chat_type,
            username,
            is_creator,
            content,
            is_mentioned,
            mentions_cristian,
            silent_mode,
        )

        if emoji:
            try:
                await asyncio.sleep(random.uniform(0.8, 2.0))
                await message.set_reaction(reaction=[ReactionTypeEmoji(emoji=emoji)])
            except Exception as e_react:
                log_error("set_reaction", e_react)

        if reply:
            delay = calculate_typing_delay(reply)
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            elapsed = 0.0
            step = 2.0
            while elapsed < delay:
                sleep_time = min(step, delay - elapsed)
                await asyncio.sleep(sleep_time)
                elapsed += sleep_time
                if elapsed < delay:
                    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

            try:
                await message.reply_text(reply)
            except Exception as e_send:
                log_error("reply_text", e_send)
                if is_creator or chat_type == "private":
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"⚠️ Error al enviar la respuesta en Telegram: {format_short_error(e_send)}",
                    )
                return

            db.save_message(chat_id, context.bot.id, "Jaison", reply, is_bot=True)

        if sticker_key and sticker_key in STICKER_CATALOG:
            try:
                await asyncio.sleep(random.uniform(1.2, 2.5))
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.CHOOSE_STICKER)
                await asyncio.sleep(random.uniform(0.8, 1.5))
                file_id = STICKER_CATALOG[sticker_key]
                await message.reply_sticker(sticker=file_id)
                db.save_message(chat_id, context.bot.id, "Jaison", f"[Sticker enviado: {sticker_key}]", is_bot=True)
            except Exception as e_stk:
                log_error(f"reply_sticker:{sticker_key}", e_stk)
                if is_creator or chat_type == "private":
                    await message.reply_text(
                        f"⚠️ No pude enviar el sticker `{sticker_key}`: {format_short_error(e_stk)}",
                        parse_mode="Markdown",
                    )

    except Exception as e_top:
        log_error("handle_message", e_top)
        try:
            if is_creator or (update.effective_chat and update.effective_chat.type == "private"):
                await message.reply_text(
                    f"⚠️ Ocurrió un error interno al procesar el mensaje: {format_short_error(e_top)}"
                )
        except Exception:
            pass


async def health_check(request):
    return web.Response(text="Jaison está activo y observando.")


async def start_web_server():
    app = web.Application()
    app.router.add_get("/", health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()


async def main():
    db.init_db()
    refresh_sticker_catalog()
    await start_web_server()

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler(["ayuda", "help"], cmd_ayuda))
    application.add_handler(CommandHandler(["modelos", "models"], cmd_modelos))
    application.add_handler(CommandHandler(["modelo", "model"], cmd_modelo))
    application.add_handler(CommandHandler("recuerdos", cmd_recuerdos))
    application.add_handler(CommandHandler("olvidar", cmd_olvidar))
    application.add_handler(CommandHandler("silencio", cmd_silencio))
    application.add_handler(CommandHandler("hablar", cmd_hablar))
    application.add_handler(CommandHandler("estado", cmd_estado))
    application.add_handler(CommandHandler("idsticker", cmd_idsticker))
    application.add_handler(CommandHandler(["pack", "importarpack"], cmd_pack))
    application.add_handler(CommandHandler("autofav", cmd_autofav))
    application.add_handler(CommandHandler("autogrupo", cmd_autogrupo))
    application.add_handler(CommandHandler("stickers", cmd_stickers))
    application.add_handler(CommandHandler("borrarsticker", cmd_borrarsticker))

    media_and_text_filter = (
        (filters.TEXT | filters.PHOTO | filters.VIDEO | filters.VOICE | filters.AUDIO | filters.Sticker.ALL | filters.ANIMATION | filters.Document.ALL)
        & ~filters.COMMAND
    )
    application.add_handler(MessageHandler(media_and_text_filter, handle_message))

    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
