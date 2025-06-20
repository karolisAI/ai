import streamlit as st
from helpers.rag import get_rag_chain
from helpers.functions import calculate_calories, generate_workout, recommend_supplements, WorkoutGenerationError, format_calorie_response
from helpers.monitoring import log_query, validate_input, validate_embeddings, log_embedding_operation
from helpers.security import (
    validate_health_metrics, sanitize_workout_input, validate_exercise_safety,
    validate_supplement_input, validate_user_age_consent,
    check_health_warning_conditions
)
from langchain_core.tools import Tool
import os
import base64
import json
from datetime import datetime

from dotenv import load_dotenv
from helpers.document_manager import DocumentManager
from langchain_community.cache import InMemoryCache
import gc
import langchain
import logging
import re
import unicodedata

load_dotenv()
OPENAI_API = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="🏋️ AI Bodybuilding Coach")

# add security disclaimer
st.sidebar.warning("""
    ⚠️ Health & Safety Notice:
    - This app provides general fitness advice only
    - Consult healthcare providers before starting any exercise program
    - Not suitable for users under 13 years old
    - Follow proper form and safety guidelines
""")

# add after the security disclaimer in the sidebar
with st.sidebar.expander("ℹ️ How to Use This Chatbot"):
    st.markdown("""
        **Special Commands** (Type exactly as shown):
        - `workout plan` - Create a personalized workout routine
        - `calculate calories` - Calculate your daily calorie needs
        - `supplements` - Get supplement recommendations
        
        **General Questions:**
        You can also ask any fitness-related questions in natural language!
        
        **Examples:**
        - "What's the best way to build muscle?"
        - "How can I improve my squat form?"
        - "What should I eat before working out?"
        
        **Tips:**
        - Be specific in your questions
        - Include relevant details like age, weight, experience level
        - Check sources in the expandable section below answers
        
        **Safety:**
        - Always consult healthcare providers
        - Follow proper form and safety guidelines
        - Not suitable for users under 13 years of age
    """)

# add a feature using Streamlit's built-in tour features
if "show_tour" not in st.session_state:
    st.session_state.show_tour = True
    st.session_state.tour_step = 0

def handle_tour_action(action: str):
    if action == "start":
        st.session_state.tour_step = 1
        st.balloons()
    elif action == "skip":
        st.session_state.show_tour = False
    elif action == "next":
        st.session_state.tour_step += 1
    elif action == "finish":
        st.session_state.show_tour = False
        st.balloons()

if st.session_state.show_tour:
    tour = st.empty()
    with tour.container():
        if st.session_state.tour_step == 0:
            st.info("👋 Welcome to the AI Bodybuilding Coach! Let's take a quick tour of the features.")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.button("Start Tour", on_click=handle_tour_action, args=("start",))
            with col2:
                st.button("Skip Tour", on_click=handle_tour_action, args=("skip",))
        elif st.session_state.tour_step == 1:
            st.success("💬 **Chat Interface**")
            st.markdown("""
                This is where you can ask any fitness-related questions. Try:
                - "How can I build muscle?"
                - "What's a good workout for beginners?"
                - "How many calories should I eat?"
            """)
            st.button("Next", on_click=handle_tour_action, args=("next",))
        elif st.session_state.tour_step == 2:
            st.success("🎯 **Special Commands**")
            st.markdown("""
                Use these commands for specific features:
                - `workout plan` - Get a personalized workout routine
                - `calculate calories` - Calculate your daily calorie needs
                - `supplements` - Get supplement recommendations
            """)
            st.button("Next", on_click=handle_tour_action, args=("next",))
        elif st.session_state.tour_step == 3:
            st.success("📊 **Health Metrics**")
            st.markdown("""
                For personalized advice, you can provide:
                - Age
                - Weight
                - Height
                - Experience level
                - Fitness goals
            """)
            st.button("Next", on_click=handle_tour_action, args=("next",))
        elif st.session_state.tour_step == 4:
            st.success("ℹ️ **Information Sources**")
            st.markdown("""
                All answers are backed by:
                - Scientific research
                - Expert knowledge
                - Reliable fitness resources
                
                Check the sources in the expandable section below answers!
            """)
            st.button("Finish Tour", on_click=handle_tour_action, args=("finish",))

PDF_DIR = './data/Nippard_Hypertrophy/'

def create_pdf_link(pdf_path, filename):
    """Create a download link for a PDF file"""
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{filename}" target="_blank">{filename}</a>'
    return href

FITNESS_URLS = [
    "https://exrx.net/Lists/Directory",
    "https://www.acsm.org/education-resources/trending-topics-resources",
    "https://www.eatright.org/fitness/exercise",
    "https://journals.lww.com/nsca-jscr/pages/default.aspx",
    "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
    "https://fdc.nal.usda.gov/",
]

def secure_calculate_calories(age, weight, height, gender, activity_level, goal):
    """Secure wrapper for calculate_calories function"""
    metrics = {
        'age': age,
        'weight': weight,
        'height': height
    }
    
    # validate health metrics
    errors = validate_health_metrics(metrics)
    if errors:
        raise ValueError('\n'.join(errors))
    
    # check for health warnings
    warnings = check_health_warning_conditions(metrics)
    for warning in warnings:
        st.warning(warning)
    
    # validate age consent
    if not validate_user_age_consent(int(age)):
        raise ValueError("This service is not available for users under 13 years old")
    
    return calculate_calories(age, weight, height, gender, activity_level, goal)

def secure_generate_workout(muscle_group, experience_level, equipment_available):
    """Secure wrapper for generate_workout function"""
    # sanitize inputs
    muscle_group = sanitize_workout_input(muscle_group)
    experience_level = sanitize_workout_input(experience_level)
    
    # validate exercise safety
    if not validate_exercise_safety(muscle_group, experience_level):
        st.warning(f"Some exercises may be too advanced for {experience_level} level. Please ensure proper form and consider working with a trainer.")
    
    return generate_workout(muscle_group, experience_level, equipment_available)

def secure_recommend_supplements(goal, diet_preferences):
    """Secure wrapper for recommend_supplements function"""
    # check for supplement warnings
    warnings = validate_supplement_input({'goal': goal})
    for warning in warnings:
        st.warning(warning)
    
    return recommend_supplements(goal, diet_preferences)

# update tools with secure versions
tools = [
    Tool(
        name="Calculate Calories",
        func=secure_calculate_calories,
        description="Calculates daily calorie needs based on age, weight, height, gender, activity level, and goal. Input format: age, weight, height, gender, activity_level, goal"
    ),
    Tool(
        name="Generate Workout",
        func=secure_generate_workout,
        description="Generates a workout plan for a specific muscle group based on experience level and equipment availability. Input format: muscle_group, experience_level, equipment_available"
    ),
    Tool(
        name="Recommend Supplements",
        func=secure_recommend_supplements,
        description="Recommends supplements based on fitness goals and dietary preferences. Input format: goal, diet_preferences"
    )
]

# Initialize document manager
document_manager = DocumentManager()

# --- Modularize Document Upload ---
def handle_document_upload():
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt'])
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    file_path = document_manager.save_uploaded_file(uploaded_file)
                    if file_path:
                        chunks = document_manager.process_document(file_path)
                        if chunks:
                            temp_retriever = get_rag_chain("", [], max_tokens=1500)
                            from langchain_openai import OpenAIEmbeddings
                            from helpers.rag import batch_embed
                            embedder = OpenAIEmbeddings(api_key=OPENAI_API)
                            texts = [doc.page_content for doc in chunks]
                            metadatas = [doc.metadata for doc in chunks]
                            ids = [f"doc_{i}" for i in range(len(chunks))]
                            try:
                                embeddings = batch_embed(texts, embedder, batch_size=50, max_retries=3)
                                validate_embeddings(embeddings)
                                collection = temp_retriever.retriever.vectorstore._collection
                                batch_size = 50
                                for i in range(0, len(texts), batch_size):
                                    batch_texts = texts[i:i + batch_size]
                                    batch_metadatas = metadatas[i:i + batch_size]
                                    batch_ids = ids[i:i + batch_size]
                                    batch_embeddings = embeddings[i:i + batch_size]
                                    try:
                                        collection.add(
                                            documents=batch_texts,
                                            metadatas=batch_metadatas,
                                            ids=batch_ids,
                                            embeddings=batch_embeddings
                                        )
                                        log_embedding_operation(
                                            "collection_add",
                                            len(batch_texts),
                                            len(texts),
                                            success=True
                                        )
                                    except Exception as e:
                                        log_embedding_operation(
                                            "collection_add",
                                            len(batch_texts),
                                            len(texts),
                                            success=False,
                                            error=e
                                        )
                                        raise
                                st.success(f"Successfully processed and embedded {len(chunks)} chunks from the document.")
                            except Exception as e:
                                st.error(f"Error during embedding process: {str(e)}")
                                logging.error(f"Embedding process error: {str(e)}")
                                raise
                        else:
                            st.error("No text chunks were extracted from the document.")
                    else:
                        st.error("Failed to save the uploaded file.")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    logging.error(f"Document processing error: {str(e)}")
    # Show uploaded documents
    st.markdown("### Your Documents")
    documents = document_manager.get_uploaded_documents()
    if documents:
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(doc)
            with col2:
                if st.button("🗑️", key=f"delete_{doc}"):
                    if document_manager.delete_document(doc):
                        st.success("Document deleted")
                        st.rerun()
    else:
        st.info("No documents uploaded yet")

with st.sidebar.expander("📚 Upload Documents"):
    st.markdown("""
        Upload your own fitness-related documents to enhance the knowledge base.
        Supported formats: PDF, TXT
    """)
    handle_document_upload()

# Initialize cache
langchain.cache = InMemoryCache()

# Configure memory management
MEMORY_WINDOW = 5  # Number of message pairs to keep in memory
MAX_TOKENS_PER_QUERY = 1000

APP_VERSION = "v1.0"

st.sidebar.info(f"App Version: {APP_VERSION}")

@st.cache_resource(show_spinner="Initializing knowledge base...")
def setup_chain(app_version=APP_VERSION):
    return get_rag_chain(PDF_DIR, FITNESS_URLS, max_tokens=1500)

def cleanup_memory():
    """Clean up memory and cache periodically."""
    # Clear the retriever's cache
    if 'chain' in st.session_state:
        st.session_state.chain.retriever.clear_cache()
    
    # Force garbage collection
    gc.collect()

# --- Chat History Helpers ---
def get_chat_history():
    chat_history = st.session_state.get("chat_history", [])
    # Ensure it's a list of tuples
    if not isinstance(chat_history, list) or any(not isinstance(x, tuple) or len(x) != 2 for x in chat_history):
        st.session_state["chat_history"] = []
        return []
    return chat_history

def update_chat_history(question, answer):
    chat_history = get_chat_history()
    chat_history.append((question, answer))
    st.session_state["chat_history"] = chat_history

# --- Meta-question detection helpers ---
def _normalize(text: str) -> str:
    """Lower-case and strip accents for crude matching."""
    text = text.lower().strip()
    return unicodedata.normalize("NFKD", text)

def detect_history_request(text: str):
    """Return one of 'list', 'last', None based on user query."""
    t = _normalize(text)
    if "last question" in t or "previous question" in t or "did i just ask" in t:
        return "last"
    # list/show all questions i have asked
    if "question" in t and ("list" in t or "show" in t or "all" in t):
        return "list"
    return None

def answer_history_request(kind: str):
    hist = get_chat_history()
    if not hist:
        return "You haven't asked any questions yet."
    if kind == "last":
        return f'Your last question was: "{hist[-1][0]}"'
    if kind == "list":
        qs = [q for q, _ in hist]
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))
        return "Here are all your questions so far:\n" + numbered
    return "I'm not sure which part of the chat history you want."

# --- Centralized Chat Handler ---
def handle_chat_interaction(user_input):
    try:
        validated_input = validate_input(user_input)

        # Quick path for meta-questions about previous user turns
        meta_kind = detect_history_request(validated_input)
        if meta_kind:
            answer = answer_history_request(meta_kind)
            update_chat_history(validated_input, answer)
            return {"answer": answer, "result": {"source_documents": []}, "question": validated_input}

        chain = setup_chain(APP_VERSION)
        # get chat history as BaseMessage list from chain memory (or empty list)
        try:
            mem_vars = chain.memory.load_memory_variables({}) if hasattr(chain, "memory") else {}
            hist_msgs = mem_vars.get("chat_history", [])
        except Exception:
            hist_msgs = []

        input_dict = {
            "question": validated_input
        }
        print(f"[DEBUG] chain.invoke input: {input_dict}")
        result = chain.invoke(input_dict)
        if result is None:
            st.error("Chatbot is currently unavailable. Please try again later.")
            return None
        _ans = result.get("answer", result.get("result", "No answer found."))
        # Fallback logic: if no relevant context, let LLM answer from its own knowledge
        source_docs = result.get("source_documents", [])
        if not source_docs or all(not doc.page_content.strip() for doc in source_docs):
            if not _ans or _ans.strip().lower() in ["i don't know", "no answer found."]:
                _ans = "I'm sorry, I couldn't find relevant information in the knowledge base, but here's what I know: " + validated_input
        update_chat_history(validated_input, _ans)
        log_query(validated_input, _ans, source="knowledge_base")
        cleanup_memory()
        return {"answer": _ans, "result": result, "question": validated_input}
    except ValueError as ve:
        st.error(str(ve))
        log_query(user_input, f"Error: {str(ve)}", source="error")
        return None
    except Exception as e:
        st.error(f"An error occurred while fetching the response: {e}")
        log_query(user_input, f"Error: {str(e)}", source="error")
        return None

def format_chat_history(chat_history):
    """Format chat history as a readable numbered list of Q&A pairs."""
    if not chat_history:
        return ""
    lines = []
    for i, (q, a) in enumerate(chat_history, 1):
        lines.append(f"Q{i}: {q}\nA{i}: {a}")
    return "\n".join(lines)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.title("🏋️ AI Bodybuilding Coach")
st.markdown("""
    Ask me anything about:
    - Calculating your daily calorie needs
    - Getting a personalized workout plan
    - Supplement recommendations
    - General fitness and nutrition advice
    - Any other fitness-related questions
""")

with st.form("chat_form"):
    user_input = st.text_input("Ask me anything about bodybuilding, fitness, or nutrition!", key="user_input")
    submit_button = st.form_submit_button("Submit")

def handle_workout_plan_command():
    st.write("🏋️ Let's create your workout plan!")
    muscle_group = st.selectbox(
        "Which muscle group would you like to target?",
        ["Chest", "Back", "Legs", "Shoulders", "Arms", "Full Body"]
    )
    experience_level = st.selectbox(
        "What's your experience level?",
        ["Beginner", "Intermediate", "Advanced"]
    )
    equipment = st.multiselect(
        "What equipment do you have access to?",
        ["Dumbbells", "Barbell", "Cable Machine", "Bodyweight Only", "Full Gym"]
    )
    
    if st.button("Generate Workout Plan"):
        try:
            if not equipment:  # check if equipment list is empty
                st.error("Please select at least one piece of equipment")
                return
            
            workout = secure_generate_workout(muscle_group, experience_level, equipment)
            st.write("**Your Personalized Workout Plan:**")
            for i, exercise in enumerate(workout, 1):
                st.write(f"{i}. {exercise}")
            
            # add some general advice
            st.info("""
                💡 Remember to:
                - Warm up properly before starting
                - Focus on proper form
                - Rest 1-2 minutes between sets
                - Stay hydrated
                - Stop if you feel pain (not to be confused with normal exercise discomfort)
            """)
            
        except WorkoutGenerationError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

def handle_calorie_calculator_command():
    st.write("🔢 Let's calculate your daily calorie needs!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=13, max_value=100, value=30)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        activity_level = st.selectbox(
            "Activity Level",
            [
                "Sedentary",
                "Light",
                "Moderate",
                "Very Active",
                "Extra Active"
            ],
            help="""
            - Sedentary: Little or no exercise
            - Light: Exercise 1-3 times/week
            - Moderate: Exercise 3-5 times/week
            - Very Active: Exercise 6-7 times/week
            - Extra Active: Very intense exercise daily or twice daily
            """
        )
        goal = st.selectbox(
            "Goal",
            ["Maintain", "Lose Weight", "Gain Weight"],
            help="""
            - Maintain: Stay at current weight
            - Lose Weight: Lose ~0.5kg per week
            - Gain Weight: Gain ~0.5kg per week
            """
        )
    
    if st.button("Calculate Calories"):
        try:
            calories = secure_calculate_calories(age, weight, height, gender, activity_level, goal)
            
            # get formatted response with recommendations
            response = format_calorie_response(calories, goal, activity_level)
            
            # display the results
            st.markdown(response)
            
            # add a BMI calculation
            bmi = weight / ((height/100) ** 2)
            bmi_category = get_bmi_category(bmi)
            
            with st.expander("📊 Additional Information"):
                st.write(f"**BMI:** {bmi:.1f} ({bmi_category})")
                st.write(f"**BMR:** {calculate_bmr(age, weight, height, gender):.0f} calories")
                st.write("""
                **Notes:**
                - BMI is a general indicator and doesn't account for muscle mass
                - Actual needs may vary based on genetics and other factors
                - Adjust intake based on real-world results
                """)
                
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

def get_bmi_category(bmi: float) -> str:
    """Return BMI category based on value."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def calculate_bmr(age: float, weight: float, height: float, gender: str) -> float:
    """Calculate BMR using Mifflin-St Jeor Equation."""
    if gender.lower() == 'male':
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        return (10 * weight) + (6.25 * height) - (5 * age) - 161

def handle_supplement_command():
    st.write("💊 Let's find the right supplements for you!")
    
    goal = st.selectbox(
        "What's your primary fitness goal?",
        ["Muscle Gain", "Fat Loss", "Performance", "General Health"],
        help="""
        - Muscle Gain: Focus on muscle growth and recovery
        - Fat Loss: Support metabolism and preserve muscle
        - Performance: Enhance athletic performance
        - General Health: Overall wellness and nutrition support
        """
    )
    
    diet_preferences = st.multiselect(
        "Any dietary preferences or restrictions?",
        ["Vegan", "Vegetarian", "Lactose-Free", "Gluten-Free"],
        help="""
        Select any dietary restrictions that apply:
        - Vegan: No animal products
        - Vegetarian: No meat products
        - Lactose-Free: No dairy
        - Gluten-Free: No gluten
        """
    )
    
    if st.button("Get Supplement Recommendations"):
        try:
            st.markdown(secure_recommend_supplements(goal, diet_preferences))
            
            # add supplement safety information
            with st.expander("📋 Supplement Safety Guidelines"):
                st.markdown("""
                    **Before Starting Any Supplement Regimen:**
                    1. Consult your healthcare provider
                    2. Check for potential interactions with medications
                    3. Start with minimum effective doses
                    4. Monitor for any adverse reactions
                    5. Choose third-party tested products when possible
                    
                    **Storage Tips:**
                    - Keep in a cool, dry place
                    - Keep out of direct sunlight
                    - Ensure containers are properly sealed
                    - Follow expiration dates
                    
                    **General Guidelines:**
                    - Take as directed
                    - Stay hydrated
                    - Maintain a balanced diet
                    - Keep track of what you're taking
                """)
        except Exception as e:
            st.error(f"Error getting supplement recommendations: {str(e)}")

def extract_relevant_text(query, content):
    """Return the most relevant sentence or paragraph for the query from the content."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', content.strip())
    # Find the sentence with the most query word overlap
    query_words = set(query.lower().split())
    best_sentence = max(sentences, key=lambda s: len(query_words & set(s.lower().split())), default=sentences[0] if sentences else "")
    # Optionally, return the paragraph containing the best sentence
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        if best_sentence in para:
            return para.strip()
    return best_sentence.strip()

def visualize_rag_process(query, result):
    """Visualize the RAG process with source information"""
    with st.expander("🔍 View Sources and Context"):
        st.markdown("### Retrieved Context")
        _ans = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        if not source_docs or all(not doc.page_content.strip() for doc in source_docs):
            st.info("No knowledge base source used for this answer.")
        else:
            # Deduplicate by source
            seen_sources = set()
            for doc in source_docs:
                source = doc.metadata.get("source", "")
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                with st.container():
                    st.markdown(f"**Source {len(seen_sources)}:**")
                    if source.startswith("http"):
                        st.markdown(f"🌐 Web: [{source}]({source})")
                        # Hide irrelevant content for web sources
                    else:
                        st.markdown(f"📄 PDF: {source}")
                        # Show only the most relevant sentence/paragraph for PDFs
                        relevant = extract_relevant_text(query, doc.page_content)
                        st.markdown("**Relevant Content:**")
                        st.markdown(relevant or "_No relevant content_")
                    st.markdown("---")

# Add a sidebar button to clear ChromaDB cache
def clear_chromadb_cache():
    from helpers.rag import clear_chromadb

    # Drop any live chain to close open file handles BEFORE filesystem delete
    if 'chain' in st.session_state:
        del st.session_state['chain']
    # Ensure cache of setup_chain is cleared so next call rebuilds
    setup_chain.clear()

    # Clear on-disk Chroma data
    if clear_chromadb():
        st.sidebar.success("ChromaDB cache cleared successfully! Knowledge base will rebuild on next query.")
    else:
        st.sidebar.error("Error clearing ChromaDB cache – some files may still be locked.")

if st.sidebar.button("🧹 Clear ChromaDB Cache"):
    clear_chromadb_cache()

if (submit_button or user_input) and user_input:
    with st.spinner("Generating your answer..."):
        lower_input = user_input.lower().strip()
        if lower_input == "workout plan":
            handle_workout_plan_command()
        elif lower_input == "calculate calories":
            handle_calorie_calculator_command()
        elif lower_input == "supplements":
            handle_supplement_command()
        else:
            chat_result = handle_chat_interaction(user_input)
            if chat_result:
                st.write("**Answer:**")
                st.write(chat_result["answer"])
                visualize_rag_process(chat_result["question"], chat_result["result"])

# display chat history
if st.session_state["chat_history"]:
    st.subheader("Chat History")
    for i, (question, answer) in enumerate(st.session_state["chat_history"]):
        st.markdown(f"**Q{i+1}:** {question}")
        st.markdown(f"**A{i+1}:** {answer}")
        st.markdown("---")

    # add export functionality
    if st.button("💾 Export Chat History"):
        chat_export = {
            'timestamp': datetime.now().isoformat(),
            'conversations': [
                {'question': q, 'answer': a} 
                for q, a in st.session_state["chat_history"]
            ]
        }
        # convert to JSON string
        json_str = json.dumps(chat_export, indent=2)
        # create download button
        st.download_button(
            label="Download Chat History",
            file_name="chat_history.json",
            mime="application/json",
            data=json_str
        )
