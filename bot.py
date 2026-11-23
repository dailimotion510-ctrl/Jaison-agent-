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

ALLOWED_REACTIONS = {"👍", "❤️", "🔥", "😂", "😮", "😢", "👏", "🤔", "👀", "💯", "🎉", "🫡"}

STICKER_CATALOG: dict[str, str] = {}
sticker_capture_sessions: dict[int, dict] = {}

client = genai.Client(api_key=GEMINI_API_KEY)


def refresh_sticker_catalog():
    """Recarga en memoria los stickers desde STICKERS_JSON y PostgreSQL."""
    merged: dict[str, str] = {}
    raw_json = os.environ.get("STICKERS_JSON", "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                        merged[k.strip().lower().replace(" ", "_")] = v.strip()
        except Exception:
            pass

    try:
        db_stickers = db.get_all_stickers()
        merged.update(db_stickers)
    except Exception:
        pass

    STICKER_CATALOG.clear()
    STICKER_CATALOG.update(merged)


def build_system_prompt() -> str:
    stickers_list = ", ".join(sorted(STICKER_CATALOG.keys())) if STICKER_CATALOG else "ninguno configurado"
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
- También puedes enviar un sticker usando una clave exacta de este catálogo: {stickers_list} (o null si no aplica).
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
  "sticker_key": "Una clave del catálogo de stickers o null"
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
        mensaje = (
            "Guía de comandos de Jaison (Modo Creador - Cristian):\n\n"
            "• /ayuda o /help — Muestra esta lista de comandos y ejemplos de uso.\n"
            "• /estado — Revisa si Jaison está en modo autónomo o silencio, cuántos mensajes ha observado en el chat y cuántos stickers tiene guardados.\n"
            "• /recuerdos — Lista los recuerdos guardados en la base de datos para este chat.\n"
            "  Ejemplo: /recuerdos\n"
            "• /olvidar <palabra> — Elimina de la memoria los recuerdos que contengan esa palabra o frase.\n"
            "  Ejemplo: /olvidar pizza\n"
            "• /silencio — Pone a Jaison en modo observación en este chat (solo responderá con texto si lo llaman por nombre/mención o si se meten contigo).\n"
            "• /hablar — Devuelve a Jaison su participación autónoma normal en este chat.\n"
            "• /idsticker [clave] — Activa la captura del próximo sticker que envíes (en los siguientes 2 minutos). Si indicas una clave, lo guarda directo en PostgreSQL y lo activa al instante.\n"
            "  Ejemplos: /idsticker risa  |  /idsticker amenaza_venganza  |  /idsticker\n"
            "• /stickers — Muestra todas las claves de stickers que Jaison tiene listas para usar.\n"
            "• /borrarsticker <clave> — Borra una clave de sticker de PostgreSQL y actualiza el catálogo al momento.\n"
            "  Ejemplo: /borrarsticker risa"
        )
    else:
        mensaje = (
            "Soy Jaison. Puedes hablar conmigo por chat privado o mencionarme por nombre o @ en un grupo para que entre en la conversación."
            + nota_creator_id
        )

    await update.message.reply_text(mensaje)


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
    modo = "Silencio (solo menciones)" if silent else "Autónomo activo"
    await update.message.reply_text(
        f"Estado en este chat:\n• Modo: {modo}\n• Mensajes observados en historial: {total_msgs}\n• Stickers guardados: {len(STICKER_CATALOG)}"
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


async def cmd_stickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_creator_user(update.effective_user.id):
        return
    refresh_sticker_catalog()
    if not STICKER_CATALOG:
        await update.message.reply_text("Aún no tengo stickers guardados. Usa `/idsticker <clave>` y envíame uno.", parse_mode="Markdown")
        return
    claves = "\n".join([f"• `{k}`" for k in sorted(STICKER_CATALOG.keys())])
    await update.message.reply_text(f"Stickers activos en memoria/BD:\n\n{claves}", parse_mode="Markdown")


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

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.8,
            ),
        )
        decision = json.loads(response.text)
    except Exception:
        return None, None, None

    if decision.get("new_memory"):
        db.save_memory(chat_id, decision["new_memory"])

    emoji = decision.get("reaction_emoji")
    if emoji not in ALLOWED_REACTIONS:
        emoji = None

    raw_sticker = decision.get("sticker_key")
    sticker_key = raw_sticker.strip().lower() if isinstance(raw_sticker, str) else None
    if sticker_key not in STICKER_CATALOG:
        sticker_key = None

    reply_text = (decision.get("reply_text") or "").strip()
    should_reply = bool(decision.get("should_reply"))

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

    if user and is_creator_user(user.id) and message.sticker and chat_id in sticker_capture_sessions:
        session = sticker_capture_sessions.pop(chat_id)
        now = asyncio.get_running_loop().time()
        if now <= session["expires_at"]:
            file_id = message.sticker.file_id
            sticker_emoji = message.sticker.emoji or "sin emoji"
            assigned_key = session.get("key")

            if assigned_key:
                saved_key = db.upsert_sticker(assigned_key, file_id, message.sticker.emoji)
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

    content = extract_message_content(message)
    if not content:
        return

    chat_type = update.effective_chat.type
    is_creator = is_creator_user(user.id) if user else False
    username = "Cristian" if is_creator else ((user.first_name or user.username or "Usuario") if user else "Usuario")
    bot_username = context.bot.username

    db.seed_default_memories(chat_id)
    db.save_message(chat_id, user.id if user else 0, username, content, is_bot=False)

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
        except Exception:
            pass

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

        await message.reply_text(reply)
        db.save_message(chat_id, context.bot.id, "Jaison", reply, is_bot=True)

    if sticker_key and sticker_key in STICKER_CATALOG:
        try:
            await asyncio.sleep(random.uniform(1.2, 2.5))
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.CHOOSE_STICKER)
            await asyncio.sleep(random.uniform(0.8, 1.5))
            file_id = STICKER_CATALOG[sticker_key]
            await message.reply_sticker(sticker=file_id)
            db.save_message(chat_id, context.bot.id, "Jaison", f"[Sticker enviado: {sticker_key}]", is_bot=True)
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
    application.add_handler(CommandHandler("recuerdos", cmd_recuerdos))
    application.add_handler(CommandHandler("olvidar", cmd_olvidar))
    application.add_handler(CommandHandler("silencio", cmd_silencio))
    application.add_handler(CommandHandler("hablar", cmd_hablar))
    application.add_handler(CommandHandler("estado", cmd_estado))
    application.add_handler(CommandHandler("idsticker", cmd_idsticker))
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
