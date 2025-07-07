from flask import Flask, render_template, request, redirect, url_for, session, send_file
import yaml
from ai_lab_repo import LaboratoryWorkflow
import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import tempfile

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

# Default YAML location
DEFAULT_YAML = "experiment_configs/MATH_agentlab.yaml"

# Function to load YAML config
def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        return None

# Function to save YAML config
def save_config(yaml_path, config_data):
    try:
        with open(yaml_path, 'w') as file:
            yaml.dump(config_data, file)
        return True
    except Exception as e:
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize session variables
    if 'paper_generated' not in session:
        session['paper_generated'] = False
    if 'config' not in session:
        session['config'] = None
    if 'yaml_content' not in session:
        session['yaml_content'] = ""
    if 'yaml_location' not in session:
        session['yaml_location'] = DEFAULT_YAML
    if 'research_topic' not in session:
        session['research_topic'] = "Your goal is to write a comprehensive paper about Building Safe and Beneficial AI Agents based on chapter 4 of the research titled 'Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems' (arXiv ID: 2504.01990). The paper should clearly explain the key advances, highlight the main challenges, and discuss future research directions in the field of foundation agents. Use clear language, structured sections, and include relevant examples or technical details to illustrate important points."

    # Handle form submissions
    if request.method == 'POST':
        if 'load_config' in request.form:
            yaml_location = request.form.get('yaml_location', DEFAULT_YAML)
            session['config'] = load_config(yaml_location)
            print(load_config(yaml_location))
            session['yaml_location'] = yaml_location
            if session['config']:
                with open(yaml_location, 'r') as file:
                    session['yaml_content'] = file.read()
        
        elif 'save_config' in request.form:
            edited_yaml = request.form.get('edited_yaml', '')
            try:
                parsed_yaml = yaml.safe_load(edited_yaml)
                print("Parsed YAML:", parsed_yaml)
                if save_config(session['yaml_location'], parsed_yaml):
                    session['config'] = parsed_yaml
                    session['yaml_content'] = edited_yaml
                    print("this is session config",session['config'])
                    
                    # Update environment variables
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
                return render_template('index.html', error=f"Invalid YAML: {e}")
        
        elif 'refine_topic' in request.form:
            # with tempfile.NamedTemporaryFile() as temp:
            #     prompt = ChatPromptTemplate.from_template(
            #         "Refine this research topic to be more specific and actionable for academic research: {topic}"
            #     )
            #     model = ChatOpenAI(model="gpt-3.5-turbo")
            #     chain = prompt | model | StrOutputParser()
            #     refined_topic = chain.invoke({"topic": session['research_topic']})
            #     session['research_topic'] = refined_topic
            print("Refining topic with AI...")
        
        elif 'generate_paper' in request.form:
            print("GEMINI_API_KEY set to:", os.environ.get('GEMINI_API_KEY', 'Not Set'))
            print(session['config'])
            if session['config']:
                config = session['config']
                try:
                    os.environ['GEMINI_API_KEY'] = config.get('gemini-api-key','')
                    print("GEMINI_API_KEY set to:", os.environ.get('GEMINI_API_KEY', 'Not Set'))
                    lab = LaboratoryWorkflow(
                        research_topic=session['research_topic'],
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
                        agentRxiv=False
                    )
                    
                    lab.perform_research()
                    session['lab'] = {
                        'lit_review_sum': lab.phd.lit_review_sum,
                        'plan': lab.phd.plan,
                        'exp_results': lab.phd.exp_results,
                        'lab_dir': lab.lab_dir
                    }
                    session['paper_generated'] = True
                    session['pdf_path'] = f"./{lab.lab_dir}/tex/temp.pdf" if config.get('compile-latex', False) else None
                except Exception as e:
                    app.logger.error(f"Error in index route: {str(e)}")
                    return render_template('index.html', error=f"Error generating paper: {e}")

    return render_template('index.html')

@app.route('/download_pdf')
def download_pdf():
    if 'pdf_path' in session and session['pdf_path'] and os.path.exists(session['pdf_path']):
        return send_file(session['pdf_path'], as_attachment=True, download_name="report.txt")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)