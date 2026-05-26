import sys
import os
from google import genai
from dotenv import load_dotenv
from rubrics import get_rubric
from score import score_transcript
import deterministic_cache as dc

# Hardcoded sample transcript (~200 words) matching a find_a_store scenario
SAMPLE_TRANSCRIPT = """
Agent: Good afternoon, thank you for calling The Sleep Company. My name is Amit. Am I speaking with Mr. Sharma?
Customer: Yes, Amit. I am Sharma. I was searching for your mattress store location in Mumbai.
Agent: Welcome, Mr. Sharma! I will be very glad to help you find our nearest experience center. May I know which specific area in Mumbai you are calling from?
Customer: I am currently in Andheri West.
Agent: Excellent! We have a gorgeous premium experience center in Andheri West, located right near the metro station. You can touch, feel, and experience our SmartGrid mattresses there. What is your preferred date to visit the store, Mr. Sharma?
Customer: I can come this Saturday around 4 PM.
Agent: Fantastic. I have locked in your visit for our Andheri West store this Saturday at 4 PM. I will send you the exact GPS location and store contact details on WhatsApp right now to make it easy for you.
Customer: Okay, thank you. That is very helpful.
Agent: You're very welcome, Mr. Sharma! We look forward to seeing you this Saturday. Have a wonderful day ahead!
"""

def main():
    load_dotenv()
    
    # Check for --no-cache flag
    if "--no-cache" in sys.argv:
        dc.USE_CACHE = False
        print("--- RUNNING WITH CACHING DISABLED (RAW LLM VARIANCE TEST) ---")
    else:
        dc.USE_CACHE = True
        print("--- RUNNING WITH PERSISTENT CACHING ENABLED ---")

    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    rubric = get_rubric("find_a_store")
    
    scores = []
    verdicts = []
    
    print("\nStarting 5 consecutive scoring runs on the sample transcript against 'find_a_store'...\n")
    
    for i in range(1, 6):
        print(f"Run {i}/5...", end="", flush=True)
        # Clear or keep cache depending on flag
        result, model_used = score_transcript(SAMPLE_TRANSCRIPT, rubric, gemini_client)
        total_score = result.get("total_score", 0)
        
        # Extract verdicts for parameters
        run_verdicts = {}
        for param_key, param_info in result.get("parameter_scores", {}).items():
            run_verdicts[param_key] = param_info.get("verdict", "")
        # Add red flags to verdicts to verify determinism of red flags too
        for rk, rv in result.get("red_flags", {}).items():
            run_verdicts[f"rf_{rk}"] = str(rv)
            
        scores.append(total_score)
        verdicts.append(run_verdicts)
        print(f" Done. Score: {total_score} (Model: {model_used})")

    # Evaluate determinism
    all_scores_identical = len(set(scores)) == 1
    
    # Check if all sets of verdicts are identical
    all_verdicts_identical = True
    differing_parameters = set()
    
    first_verdict = verdicts[0]
    for i in range(1, 5):
        current_verdict = verdicts[i]
        if current_verdict != first_verdict:
            all_verdicts_identical = False
            for k in first_verdict:
                if first_verdict[k] != current_verdict[k]:
                    differing_parameters.add(k)
                    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY:")
    print("=" * 60)
    for i in range(5):
        print(f"Run {i+1}: Total Score = {scores[i]}")
        
    print("-" * 60)
    if all_scores_identical and all_verdicts_identical:
        print("RESULT: DETERMINISTIC")
        print("All 5 runs produced identical total scores and parameter/red-flag verdicts!")
    else:
        print("RESULT: NON-DETERMINISTIC")
        if not all_scores_identical:
            print(f"Scores varied: {scores}")
        if not all_verdicts_identical:
            print(f"Verdicts varied on parameters/red-flags: {sorted(list(differing_parameters))}")
            # Print a comparison table
            print("\nComparison of varying parameters:")
            for p in sorted(list(differing_parameters)):
                vals = [v.get(p, "") for v in verdicts]
                print(f"  {p}: {vals}")
                
    print("=" * 60)

if __name__ == "__main__":
    main()
