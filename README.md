# 📚🤖 Private Teacher Agent 

## 📝 Overview
**Private Teacher Agent** is an AI-powered tutoring assistant that conducts interactive, personalized lessons in **Math, Science, History, and SAT preparation**.  
It uses a **main ReAct agent** powered by Azure OpenAI and a collection of **specialized sub-agents/tools** to generate questions, evaluate answers, guide students step-by-step, provide motivational feedback, and track progress across sessions.

The agent adapts its teaching style to the student’s level, uses past performance data to target weaknesses, and runs lessons in a warm, concise, and encouraging manner.

---

## ✨ Features

- **Course selection** — supports `Math`, `Science`, `History`, and `SAT`.
- **Adaptive question generation** — picks suitable topics and difficulty using past evaluation notes.
- **Interactive evaluation** — grades answers with constructive feedback.
- **Step-by-step tutoring** — optional guided solving with error-specific hints.
- **Motivational nudges** — short, encouraging messages to keep the student engaged.
- **Final feedback** — summarizes session performance with concrete next steps.
- **Student progress tracking** — updates course status with a mastery score and notes.

---

## 🏗️ System Architecture

### Main Flow (`src/agent/run.py`)
1. **Collect student info**  
   - Name and ID.
   - Chosen course (validated against allowed courses).
   - Initial learning request.
2. **Initialize Main Agent**  
   - Calls `init_private_teacher()` from `src/agent/main_private_teacher.py`.
3. **Run interactive loop**  
   - Agent generates questions, evaluates answers, and adapts the lesson dynamically.

### Main Agent (`src/agent/main_private_teacher.py`)
- Uses `langchain` ReAct agent with Azure OpenAI.
- **Tools:**
  - `generate_question_agent` — Creates course-specific questions using RAG from DB/web sources.
  - `present_message_to_user` — Displays messages and gets student responses.
  - `evaluate_answer` — Grades answers and records common mistakes.
  - `hand_in_hand_agent` — Guides the student step-by-step.
  - `get_coacher_response` — Sends motivational messages.
  - `provide_final_feedback` — Summarizes the session.
      -  Uses an additional agent tool `student_evaluator/update_student_course_status` - Updates the student's course status and progress.

### Prompt Library (`prompts.py`)
- **Course prompts** — Guidelines for structuring questions in Math, Science, History, SAT.
- **System/User prompts** — For question generation, coaching, grading, updating student status, and interactive tutoring.
- **Lesson orchestration** — Ensures the Main Private Teacher stays in persona, manages flow, and controls when tools are called.

---

## 💻 Installation

```bash
# Clone repository
git clone <repo-url>
cd PrivateTeacherAgent

# Install dependencies
pip install -r requirements.txt
```

You also need to configure your `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
```

---

## ▶️ Usage

Run the private teacher agent:

```bash
python src/agent/run.py
```

Example interaction:

```
📚🤖 AI Teacher:
Hi! What is your name?

🎓 Student: Ben

📚🤖 AI Teacher:
Nice to meet you, Ben! Please enter an id number:

🎓 Student: 12345

📚🤖 AI Teacher:

Hi Ben! 👋 I’m your private teacher for today! Ready to help you learn, practice, and improve.
My main areas of expertise are Math, History, Science, and SAT questions.

What course would you like to focus on today?
Please enter one of the following options: Math, History, Science, SAT

🎓 Student: Math

📚🤖 AI Teacher:

That's great Ben! Lets study today a bit of Math.
Do you have any specific topics or requests in mind?
...
```
For full size examples, please refer to the directory `examples`.

### Debug
Change config in `src/utils/constants.py` and run again.
```
DEBUG_MODE=True
```


---

## 📂 Project Structure
- examples/                       
- src/
   - agent/
      - answer_evaluator.py
      - coacher.py
      - general_feedback_generator.py
      - hand_in_hand_solver.py      
      - main_private_teacher.py    
      - prompts.py             
      - question_RAG.py            
      - run.py                   
      - student_evaluator.py      
   - data/
      - DB_questions/             
      - History/
         - Math/
         - SAT/
         - Science/
      - index_and_search.py         
   - utils/
- tokens_count/            


## 🗂️ Additional Description Per Path
```
### `src/agent/`
| Path | Description |
|------|-------------|
| `answer_evaluator.py` | Answer grading logic |
| `coacher.py` | Motivational message generator |
| `general_feedback_generator.py` | Generates final session feedback |
| `hand_in_hand_solver.py` | Step-by-step tutoring agent |
| `main_private_teacher.py` | Main ReAct agent setup and tools |
| `prompts.py` | Prompt templates for all sub-agents/tools |
| `question_RAG.py` | Question generator with DB/Web search |
| `run.py` | Entry point for running the private teacher |
| `student_evaluator.py` | Updates student course status |

### `src/data/`
| Path | Description |
|------|-------------|
| `DB_questions/` | Data and preprocessing before index to Qdrant |
| `History/` | History-related DB content |
| `Math/` | Math-related DB content |
| `SAT/` | SAT-related DB content |
| `Science/` | Science-related DB content |
| `index_and_search.py` | Indexing and searching utilities for questions |


