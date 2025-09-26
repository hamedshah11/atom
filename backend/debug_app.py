import os, sys
from pathlib import Path
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Business Idea Analyzer - DEBUG", page_icon="🐛", layout="wide")

# ---- Secrets → env BEFORE imports that use them ----
if "openai_api_key" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.title("🐛 Debug Mode - Business Idea Analyzer")

# Debug info section
with st.expander("🔍 Debug Information", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environment")
        api_key = os.getenv("OPENAI_API_KEY", "")
        st.write(f"API Key Set: {'✅' if api_key else '❌'}")
        if api_key:
            st.write(f"Key Preview: {api_key[:8]}...{api_key[-4:]}")
        st.write(f"Model: {os.getenv('MODEL_ALL', 'gpt-3.5-turbo')}")
        st.write(f"Debug Mode: {os.getenv('DEBUG_MODE', '1')}")
    
    with col2:
        st.subheader("Quick API Test")
        if st.button("Test API Connection"):
            if not api_key:
                st.error("No API key set!")
            else:
                try:
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Say 'API Working'"}],
                        max_tokens=10
                    )
                    result = response.choices[0].message.content
                    st.success(f"✅ API Response: {result}")
                except Exception as e:
                    st.error(f"❌ API Error: {e}")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ No API key detected!")
    key = st.text_input("Enter OpenAI API Key", type="password")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        st.rerun()

# Test individual agents
st.header("Test Individual Agents")

test_idea = st.text_area("Test Business Idea", 
                        value="An online platform for booking padel courts",
                        height=80)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Test Planner Agent"):
        with st.spinner("Testing planner..."):
            try:
                # Enable debug mode
                os.environ["DEBUG_MODE"] = "1"
                from backend import agents
                
                # Capture stdout for debug messages
                import io
                import contextlib
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    result = agents.planner_agent(test_idea)
                
                debug_output = f.getvalue()
                
                st.subheader("Debug Output")
                st.code(debug_output)
                
                st.subheader("Agent Result")
                if result:
                    st.success("Got response!")
                    st.write(result)
                else:
                    st.error("Empty response!")
                    
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

with col2:
    if st.button("Test Raw API Call"):
        with st.spinner("Testing raw API..."):
            try:
                client = OpenAI()
                
                # Your exact agent format
                messages = [
                    {"role": "system", "content": "You are a planning agent. Create a plan."},
                    {"role": "user", "content": f"Business idea: {test_idea}"}
                ]
                
                st.write("Sending messages:")
                st.json(messages)
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=200,
                    temperature=0.7
                )
                
                st.write("Response object type:", type(response))
                st.write("Has choices:", hasattr(response, 'choices'))
                st.write("Number of choices:", len(response.choices) if hasattr(response, 'choices') else 0)
                
                if response.choices:
                    choice = response.choices[0]
                    st.write("Choice type:", type(choice))
                    st.write("Has message:", hasattr(choice, 'message'))
                    
                    if hasattr(choice, 'message'):
                        message = choice.message
                        st.write("Message type:", type(message))
                        st.write("Has content:", hasattr(message, 'content'))
                        st.write("Content:", message.content)
                        st.write("Content is None:", message.content is None)
                        st.write("Content type:", type(message.content))
                
                st.json(response.model_dump())
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

with col3:
    if st.button("Test Simple Function"):
        st.write("Testing basic Python string operations...")
        
        # Test the exact logic from agents.py
        test_values = [None, "", "  ", "Hello", "  Hello  "]
        
        for val in test_values:
            st.write(f"\nTesting: {repr(val)}")
            st.write(f"- val.strip() if val else '': {repr(val.strip() if val else '')}")
            st.write(f"- bool(val and val.strip()): {bool(val and val.strip())}")

# Full debug run
st.header("Full Debug Run")
if st.button("Run Full Analysis with Debug", type="primary"):
    os.environ["DEBUG_MODE"] = "1"
    
    # Import after setting debug mode
    from backend import agents
    
    with st.spinner("Running with full debug output..."):
        import io
        import contextlib
        
        f = io.StringIO()
        try:
            with contextlib.redirect_stdout(f):
                st.write("### Starting Planner Agent")
                plan = agents.planner_agent(test_idea)
                
                st.write("### Planner Result:")
                st.code(plan)
                
                st.write("### Debug Logs:")
                st.code(f.getvalue())
                
        except Exception as e:
            st.error(f"Error: {e}")
            st.write("### Debug output so far:")
            st.code(f.getvalue())
            st.exception(e)
