import os
import re
import json
import random
import asyncio
import traceback
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
LAST_SCAN_ERRORS: list[str] = []

# chat_id -> {"expires_at": float, "key": str | None, "mode": "single" | "pack"}
sticker_capture_sessions: dict[int, dict] = {}

client = genai.Client(api_key=GEMINI_API_KEY)


def log_error(context_tag: str, err: Exception | str):
    """Registra errores en los logs de Render de forma visible."""
    print(f"[ERROR - {context_tag}]: {err}", flush=True)


def format_detailed_error(err: Exception | str, include_tb: bool = False, max_len: int = 3200) -> str:
    """Devuelve el error exacto con su tipo de excepción y detalle sin recortar lo importante."""
    if isinstance(err, BaseException):
        exc_type = type(err).__name__
        exc_msg = str(err).strip() or "(sin mensaje)"
        base = f"{exc_type}: {exc_msg}"
        if include_tb:
            tb_text = "".join(traceback.format_exception(type(err), err, err.__traceback__)).strip()
            combined = f"{base}\n\nTraceback:\n{tb_text}"
        else:
            combined = base
    else:
        combined = str(err).strip() or "Error desconocido"

    if len(combined) > max_len:
        return combined[: max_len - 15] + "\n...[truncado]"
    return combined


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
    LAST_SCAN_ERRORS.clear()
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
        err_async_str = format_detailed_error(e_async, include_tb=False)
        LAST_SCAN_ERRORS.append(f"aio.models.list -> {err_async_str}")
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
            err_sync_str = format_detailed_error(e_sync, include_tb=False)
            LAST_SCAN_ERRORS.append(f"sync.models.list -> {err_sync_str}")
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
                LAST_SCAN_ERRORS.append(f"ping({candidate}) -> {format_detailed_error(e_ping)}")
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


async def send_error_to_creator(context: ContextTypes.DEFAULT_TYPE, title: str, detail: str):
    """Envía el error exacto por Telegram al chat del creador si CRISTIAN_USER_ID está configurado."""
    if not CRISTIAN_USER_ID:
        return
    try:
        text = f"⚠️ Error ({title}):\n{detail}"
        if len(text) > 3900:
            text = text[:3890] + "\n...[cortado]"
        await context.bot.send_message(chat_id=CRISTIAN_USER_ID, text=text)
    except Exception as e_notify:
        log_error("send_error_to_creator", e_notify)


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
            models = await scan_available
