import streamlit as st
import yaml
from ai_lab_repo import LaboratoryWorkflow
import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import shutil

# Constants
DEFAULT_YAML = "experiment_configs/MATH_agentlab.yaml"
RESEARCH_DIR_PATH = "MATH_research_dir"
# Function to remove directory and its contents
def remove_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)

# Function to create research directory structure
def create_research_dirs(lab_index=0, paper_index=0):
    # Remove existing directory and create fresh structure
    remove_directory(RESEARCH_DIR_PATH)
    os.makedirs(RESEARCH_DIR_PATH, exist_ok=True)
    
    # Create specific lab directory
    lab_dir = os.path.join(RESEARCH_DIR_PATH, f"research_dir_lab{lab_index}_paper{paper_index}")
    os.makedirs(lab_dir, exist_ok=True)
    os.makedirs(os.path.join(lab_dir, "src"), exist_ok=True)
    os.makedirs(os.path.join(lab_dir, "tex"), exist_ok=True)
    
    # Return the path in a consistent format
    return os.path.normpath(lab_dir)  # This normalizes the path separators

# Set up the Streamlit app
st.title("Research Paper Generator")
st.write("Configure and generate research papers using AI agents")

# Initialize session state
if 'paper_generated' not in st.session_state:
    st.session_state.paper_generated = False
if 'config' not in st.session_state:
    st.session_state.config = None
if 'yaml_content' not in st.session_state:
    st.session_state.yaml_content = ""

# Default YAML location
DEFAULT_YAML = "experiment_configs/MATH_agentlab.yaml"

# Function to load YAML config
def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading YAML config: {e}")
        return None

# Function to save YAML config
def save_config(yaml_path, config_data):
    try:
        with open(yaml_path, 'w') as file:
            yaml.dump(config_data, file)
        st.success("Configuration saved successfully!")
    except Exception as e:
        st.error(f"Error saving YAML config: {e}")

# Sidebar for YAML file selection
st.sidebar.header("Configuration Management")
yaml_location = st.sidebar.text_input("YAML config location", DEFAULT_YAML)

# Load the YAML config
if st.sidebar.button("Load Config"):
    st.session_state.config = load_config(yaml_location)
    if st.session_state.config:
        with open(yaml_location, 'r') as file:
            st.session_state.yaml_content = file.read()


# Edit YAML configuration
st.header("Edit Configuration")
if st.session_state.config:
    # Display editable YAML
    edited_yaml = st.text_area(
        "Edit YAML configuration",
        st.session_state.yaml_content,
        height=600
    )

    # Save button
    if st.button("Save Configuration"):
        try:
            # Validate the YAML before saving
            parsed_yaml = yaml.safe_load(edited_yaml)
            save_config(yaml_location, parsed_yaml)
            st.session_state.config = parsed_yaml
            st.session_state.yaml_content = edited_yaml

            # Update environment variables when saving
            if 'gemini-api-key' in parsed_yaml:
                os.environ['GEMINI_API_KEY'] = parsed_yaml['gemini-api-key']
            if 'api-key' in parsed_yaml:
                os.environ['OPENAI_API_KEY'] = parsed_yaml['api-key']
            if 'deepseek-api-key' in parsed_yaml:
                os.environ['DEEPSEEK_API_KEY'] = parsed_yaml['deepseek-api-key']
            if 'groq-api-key' in parsed_yaml:
                os.environ['GROQ_API_KEY'] = parsed_yaml['groq-api-key']
            if 'openrouter-api-key' in parsed_yaml:
                os.environ['OPENROUTER_API_KEY'] = parsed_yaml['openrouter-api-key']
        except yaml.YAMLError as e:
            st.error(f"Invalid YAML: {e}")

# Research paper generation section
st.header("Generate Research Paper")
research_topic = "Your goal is to write a comprehensive paper about Building Safe and Beneficial AI Agents based on chapter 4 of the research titled 'Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems' (arXiv ID: 2504.01990). The paper should clearly explain the key advances, highlight the main challenges, and discuss future research directions in the field of foundation agents. Use clear language, structured sections, and include relevant examples or technical details to illustrate important points."
# Only show generation form if config is loaded
if st.session_state.config:
    config = st.session_state.config
    
    # Display current research topic
    research_topic = st.text_area(
        "Research Topic",
        config.get('research-topic', ''),
        key="research_topic_input"
    )
    
    # LangChain topic refinement
    if st.button("Refine Topic with AI"):
        with st.spinner("Refining topic..."):
            prompt = ChatPromptTemplate.from_template(
                "Refine this research topic to be more specific and actionable for academic research: {topic}"
            )
            model = ChatOpenAI(model="gpt-3.5-turbo")
            chain = prompt | model | StrOutputParser()
            refined_topic = chain.invoke({"topic": research_topic})
            st.session_state.research_topic = refined_topic
            st.rerun()
    
    # Generate paper button
    if st.button("Generate Research Paper"):
        with st.spinner("Generating research paper..."):
            try:
                # Create research directories
                lab_dir = create_research_dirs(
                    lab_index=config.get('lab-index', 0),
                    paper_index=0  # You can modify this if you need multiple papers
                )
            
                print(research_topic)
                os.environ['GEMINI_API_KEY'] = config.get('gemini-api-key','')
                # Create lab instance with all YAML parameters
                lab = LaboratoryWorkflow(
                    research_topic = research_topic,
                    notes=[{"phases": [k.replace("-", " ")], "note": v} 
                          for k, vs in config.get('task-notes', {}).items() 
                          for v in vs],
                    agent_model_backbone={
                        "literature review": config.get('lit-review-backend', 'gpt-4'),
                        "plan formulation": config.get('llm-backend', 'gpt-4'),
                        "data preparation": config.get('llm-backend', 'gpt-4'),
                        "running experiments": config.get('llm-backend', 'gpt-4'),
                        "results interpretation": config.get('llm-backend', 'gpt-4'),
                        "report writing": config.get('llm-backend', 'gpt-4'),
                        "report refinement": config.get('llm-backend', 'gpt-4')
                    },
                    human_in_loop_flag={phase: config.get('copilot-mode', False) 
                                      for phase in [
                                          "literature review", "plan formulation", 
                                          "data preparation", "running experiments",
                                          "results interpretation", "report writing",
                                          "report refinement"
                                      ]},
                    openai_api_key=config.get('api-key', ''),
                    compile_pdf=config.get('compile-latex', False),
                    num_papers_lit_review=config.get('num-papers-lit-review', 5),
                    papersolver_max_steps=config.get('papersolver-max-steps', 5),
                    mlesolver_max_steps=config.get('mlesolver-max-steps', 3),
                    paper_index=0,
                    except_if_fail=config.get('except-if-fail', False),
                    lab_index=config.get('lab-index', 0),
                    agentRxiv=False,
                    lab_dir=lab_dir
                )
                
                # Run the research workflow
                lab.perform_research()
                
                # Store results in session state
                st.session_state.lab = lab
                st.session_state.paper_generated = True
                
            except Exception as e:
                st.error(f"Error generating paper: {e}")
                st.error(f"research_topic: {research_topic}")
                st.session_state.paper_generated = False

    # Display results if paper was generated
    if st.session_state.get('paper_generated', False):
        st.success("Research paper generated successfully!")
        lab = st.session_state.lab
        
        # Show sections of the paper
        st.subheader("Literature Review")
        st.write(lab.phd.lit_review_sum)
        
        st.subheader("Research Plan")
        st.write(lab.phd.plan)
        
        st.subheader("Experimental Results")
        st.write(lab.phd.exp_results)
        
        # Add download button for the PDF if compiled
        if config.get('compile-latex', False):
            pdf_path = f"./{lab.lab_dir}/tex/temp.pdf"
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name="research_paper.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("PDF compilation was enabled but file not found.")
else:
    st.warning("Please load a configuration file first using the sidebar.")

# Instructions section
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Load your YAML config file
2. Edit the configuration as needed
3. Save your changes
4. Generate your research paper
""")