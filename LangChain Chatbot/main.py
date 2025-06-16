import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from helpers.rag import get_combined_retriever
from helpers.functions import calculate_calories, generate_workout, recommend_supplements, WorkoutGenerationError, format_calorie_response
from helpers.monitoring import log_query, validate_input, rate_limiter, validate_embeddings, log_embedding_operation
from helpers.security import (
    validate_health_metrics, sanitize_workout_input, validate_exercise_safety,
    validate_supplement_input, log_security_event, validate_user_age_consent,
    check_health_warning_conditions
)
from langchain_core.tools import Tool
import os
import base64
import json
from datetime import datetime

from dotenv import load_dotenv
from helpers.document_manager import DocumentManager
from helpers.baseretriever import EnhancedRetriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.memory import BaseMemory
from langchain_community.cache import InMemoryCache
from pydantic import BaseModel, Field
import gc
import langchain
from typing import Any, Dict, List
import logging

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

# Add document upload section in sidebar
with st.sidebar.expander("📚 Upload Documents"):
    st.markdown("""
        Upload your own fitness-related documents to enhance the knowledge base.
        Supported formats: PDF, TXT
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt'])
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Save the file
                    file_path = document_manager.save_uploaded_file(uploaded_file)
                    if file_path:
                        # Process the document
                        chunks = document_manager.process_document(file_path)
                        if chunks:
                            # Initialize a new retriever just for this document
                            temp_retriever = get_combined_retriever("", [], max_tokens=1500)
                            
                            # Precompute embeddings for the chunks
                            from langchain_openai import OpenAIEmbeddings
                            from helpers.rag import batch_embed
                            
                            embedder = OpenAIEmbeddings(api_key=OPENAI_API)
                            texts = [doc.page_content for doc in chunks]
                            metadatas = [doc.metadata for doc in chunks]
                            ids = [f"doc_{i}" for i in range(len(chunks))]
                            
                            try:
                                # Use the improved batch_embed function with retries
                                embeddings = batch_embed(texts, embedder, batch_size=50, max_retries=3)
                                
                                # Validate embeddings before adding to collection
                                validate_embeddings(embeddings)
                                
                                # Add to Chroma collection directly
                                collection = temp_retriever._base_retriever.vectorstore._collection
                                
                                # Add documents in smaller batches to avoid memory issues
                                batch_size = 50  # Use smaller batch size for collection addition
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

# Initialize cache
langchain.cache = InMemoryCache()

# Configure memory management
MEMORY_WINDOW = 5  # Number of message pairs to keep in memory
MAX_TOKENS_PER_QUERY = 1000

class WindowedMemory(BaseMemory, BaseModel):
    messages: List[dict] = Field(default_factory=list)
    k: int = Field(default=5)
    return_messages: bool = Field(default=True)
    output_key: str = Field(default="answer")
    input_key: str = Field(default="question")
    memory_key: str = Field(default="chat_history")

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {self.memory_key: self.messages[-self.k * 2:] if self.messages else []}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        if self.input_key not in inputs:
            raise ValueError(f"input key {self.input_key} not found in inputs")
        
        # Save user message
        self.messages.append({
            "type": "human",
            "content": inputs[self.input_key]
        })
        
        # Save AI message
        if self.output_key in outputs:
            self.messages.append({
                "type": "ai",
                "content": outputs[self.output_key]
            })
        
        # Maintain window size
        if len(self.messages) > self.k * 2:
            self.messages = self.messages[-self.k * 2:]

    def clear(self) -> None:
        self.messages = []

@st.cache_resource(show_spinner="Initializing knowledge base...")
def setup_chain():
    return get_combined_retriever(PDF_DIR, FITNESS_URLS, max_tokens=1500)

def cleanup_memory():
    """Clean up memory and cache periodically."""
    # Clear the retriever's cache
    if 'chain' in st.session_state:
        st.session_state.chain.retriever.clear_cache()
    
    # Force garbage collection
    gc.collect()

# Add memory cleanup to the main processing function
def process_with_knowledge_base(input_text):
    """Process user input with knowledge base and memory management."""
    try:
        # Validate input
        validate_input(input_text)
        
        # Process with rate limiting
        @rate_limiter
        def process():
            chain = setup_chain()
            result = chain.invoke({"question": input_text})
            
            # Log the query
            log_query(input_text, result)
            
            # Clean up memory periodically
            cleanup_memory()
            
            return result
        
        return process()
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        log_security_event("error", str(e))
        return None

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
            recommendations = secure_recommend_supplements(goal, diet_preferences)
            st.markdown(recommendations)
            
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

def visualize_rag_process(query, result):
    """Visualize the RAG process with source information"""
    with st.expander("🔍 View Sources and Context"):
        st.markdown("### Retrieved Context")
        
        # Display source documents
        for i, doc in enumerate(result.get("source_documents", []), 1):
            with st.container():
                st.markdown(f"**Source {i}:**")
                source_type = doc.metadata.get("source_type", "unknown")
                if source_type == "web":
                    st.markdown(f"🌐 Web: {doc.metadata.get('url', 'Unknown URL')}")
                elif source_type == "pdf":
                    st.markdown(f"📄 PDF: {doc.metadata.get('file_name', 'Unknown File')}")
                
                st.markdown("**Relevant Content:**")
                st.markdown(doc.page_content)
                st.markdown("---")

if (submit_button or user_input) and user_input:
    if process_with_knowledge_base(user_input) is None:
        st.error("Chatbot is currently unavailable. Please try again later.")
    else:
        with st.spinner("Generating your answer..."):
            try:
                # check for special commands first
                lower_input = user_input.lower().strip()
                if lower_input == "workout plan":
                    handle_workout_plan_command()
                elif lower_input == "calculate calories":
                    handle_calorie_calculator_command()
                elif lower_input == "supplements":
                    handle_supplement_command()
                else:
                    # proceed with the existing general query handling
                    validated_input = validate_input(user_input)
                    
                    # first try to use the tools
                    try:
                        @rate_limiter
                        def process_with_tools(input_text):
                            return process_with_tools(input_text)
                        
                        response = process_with_tools(validated_input)
                        log_query(validated_input, response, source="tools")
                        st.session_state["chat_history"].append((validated_input, response))
                        st.write("**Answer:**")
                        st.write(response)
                    except Exception as agent_error:
                        # if tools can't handle it, use the knowledge base
                        @rate_limiter
                        def process_with_knowledge_base(input_text):
                            return process_with_knowledge_base(input_text)
                        
                        result = process_with_knowledge_base(validated_input)
                        log_query(validated_input, result, source="knowledge_base")
                        st.session_state["chat_history"].append((validated_input, result))
                        
                        st.write("**Answer:**")
                        st.write(result)
                        
                        # display source URLs
                        visualize_rag_process(validated_input, result)

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"An error occurred while fetching the response: {e}")
                log_query(validated_input, f"Error: {str(e)}", source="error")

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
