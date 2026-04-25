import streamlit as st
import requests
import re
import urllib.parse
import xml.etree.ElementTree as ET
from groq import Groq

# ==========================================
# ⚙️ CONFIGURATION CONSTANTS
# ==========================================
MAX_HISTORY_TURNS = 6
MAX_WEB_RESULTS   = 3
REQUEST_TIMEOUT_SEC = 5

# ==========================================
# 🧠 THE SOVEREIGN ENGINE (v24.1 MASTER)
# ==========================================
class JarvisSingularity:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.router_model = "llama-3.1-8b-instant"
        self.core_model = "llama-3.3-70b-versatile"

    def _classify_intent(self, query: str) -> bool:
        """
        Decides if a query needs live internet data.

        Args:
            query: The user's question.

        Returns:
            True if web search is needed, False if not.
        """
        prompt = f"""
        Does the following query require live internet search, current news, real-time prices, or modern facts to answer accurately?
        Query: "{query}"
        Answer strictly with the word YES or the word NO. No other text.
        """
        try:
            res = self.client.chat.completions.create(
                model=self.router_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            ).choices[0].message.content.strip().upper()
            return "YES" in res
        except Exception as e:
            st.warning("⚠️ Router check failed, skipping web search.")
            return False

    def _fetch_web_data(self, query: str) -> str:
        """
        Fetches live web data using two fallback nodes.

        Args:
            query: The user's question to search for online.

        Returns:
            A string of search results, or error message if both nodes fail.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        # NODE 1: Live News RSS Stream
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
            root = ET.fromstring(response.content)

            news_items = []
            for item in root.findall('.//item')[:MAX_WEB_RESULTS]:
                title = item.find('title').text
                pubDate = item.find('pubDate').text
                news_items.append(f"- {title} (Published: {pubDate})")

            data = "\n".join(news_items)
            if data:
                return data
        except Exception:
            pass  # Silently drop to Node 2

        # NODE 2: Wikipedia API Fallback
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&utf8=&format=json"
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC).json()
            results = response['query']['search'][:MAX_WEB_RESULTS]

            clean_data = []
            for r in results:
                clean_snippet = re.sub(r'<[^>]+>', '', r['snippet'])
                clean_data.append(f"- {r['title']}: {clean_snippet}")

            data = "\n".join(clean_data)
            if data:
                return data
        except Exception as e:
            return f"SYSTEM ALERT: Absolute Web Severance. ({str(e)})"

        return "No recent data found."

    def generate_response(self, query: str, history: list):
        """
        Generates a streaming response from Jarvis.

        Args:
            query: The user's message.
            history: List of past chat messages.

        Yields:
            Words of the response one chunk at a time.
        """
        needs_web = self._classify_intent(query)
        live_context = ""

        if needs_web:
            with st.spinner("🌐 Accessing Live 2026 Data Streams..."):
                live_context = self._fetch_web_data(query)

        system_prompt = """
        You are J.A.R.V.I.S., an elite, highly intelligent, and conversational AI assistant.

        CORE DIRECTIVES:
        1. Tone: Crisp, polite, British professionalism. Address the user as 'Sir'.
        2. Format: Give straight, direct answers. Do not use filler words or robotic disclaimers.
        3. Logic: Be highly analytical, yet converse normally like a brilliant human partner.
        4. Context: If LIVE WEB DATA is provided below, use it to ground your answer in the present day.
        """

        if live_context:
            system_prompt += f"\n\n[LIVE WEB DATA FOR QUERY]:\n{live_context}"

        messages = [{"role": "system", "content": system_prompt}]

        for msg in history[-MAX_HISTORY_TURNS:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})

        try:
            stream = self.client.chat.completions.create(
                model=self.core_model,
                messages=messages,
                temperature=0.6,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"Sir, I encountered a critical system error: {str(e)}"


# ==========================================
# ⚙️ INTERFACE: J.A.R.V.I.S. TERMINAL
# ==========================================
st.set_page_config(page_title="J.A.R.V.I.S. APEX", layout="wide")
st.markdown("<style>.stApp { background-color: #050505; color: #00FFCC; }</style>", unsafe_allow_html=True)

@st.cache_resource
def get_jarvis():
    return JarvisSingularity(api_key=st.secrets["GROQ_API_KEY"])

jarvis = get_jarvis()

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# ==========================================
# 🎛️ J.A.R.V.I.S. COMMAND CENTER (SIDEBAR)
# ==========================================
with st.sidebar:
    st.title("🧿 SYSTEM TELEMETRY")
    st.markdown("---")

    st.write(f"**🧠 Core Engine:** `{jarvis.core_model}`")
    st.write(f"**⚡ Router Node:** `{jarvis.router_model}`")
    st.write("**🌐 Web Matrix:** `Online (Dual-Node)`")
    st.write("**🔒 Security:** `Vault Protocol Active`")

    st.markdown("---")
    st.subheader("Memory Management")

    if st.button("🧹 Purge Short-Term Memory", use_container_width=True):
        st.session_state.chat_log = []
        st.rerun()

    if st.session_state.chat_log:
        chat_text = "# J.A.R.V.I.S. Neural Dump\n\n"
        for msg in st.session_state.chat_log:
            role = "USER" if msg["role"] == "user" else "J.A.R.V.I.S."
            chat_text += f"### {role}\n{msg['content']}\n\n---\n\n"

        st.download_button(
            label="📥 Export Session Log (.md)",
            data=chat_text,
            file_name="JARVIS_Neural_Dump.md",
            mime="text/markdown",
            use_container_width=True
        )

# ==========================================
# 💬 MAIN CHAT RENDER LOOP
# ==========================================
for msg in st.session_state.chat_log:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Command the Matrix, Sir..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = st.write_stream(
            jarvis.generate_response(prompt, st.session_state.chat_log)
        )

    st.session_state.chat_log.append({"role": "user", "content": prompt})
    st.session_state.chat_log.append({"role": "assistant", "content": answer})
