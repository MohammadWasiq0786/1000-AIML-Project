from agents.planner import run_planner
from agents.researcher import run_researcher
from agents.writer import run_writer
from agents.editor import run_editor
 
def run_pipeline():
    run_planner()
    run_researcher()
    run_writer()
    run_editor()
 
if __name__ == "__main__":
    run_pipeline()
    print("Book written! Check output/draft.txt")