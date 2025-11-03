from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

sample_emails = [
    "I've been charged twice for my subscription this month. Can you refund one payment?",
    "The app crashes every time I try to upload a file larger than 10MB.",
    "I'm interested in upgrading to the enterprise plan. What features does it include?"
]

def zero_shot_classification(email):
    """
    No examples provided. Just intructions.
    Good for : Simple, well-defined tasks where the model already understands the domain. 
    """
    prompt = f""" You are a customer support assistant. Classify the following email into one of these categories:
       1. Billing Issue
       2. Technical Problem
       3. Feature Request
       4. Sales
       5. General Inquiry

       Email: {email}

       Category:"""

    response = client.chat.completions.create(
               model="openai/gpt-oss-120b",
               messages=[{"role": "user", "content": prompt}],
               temperature=0.1,
               max_tokens=50
               )

    return response.choices[0].message.content.strip()

def few_shot_classification(email):
    """
    Give example outputs along with the input in the prompt.
    Best for: Consistent, reliable and predictable outputs.
    Accuracy: 3-5 examples: 80-90% 
              5-10 examples: 95-98%   
              10+ diminishing returns
    Note: More tokens, more costly than zero-shot but more accurate than zero shot.
    """

    prompt = f""" Classify customer emails into categories based on these examples:

Email: "My credit card was declined but I was still charged."
Category: Billing

Email: "I want to inquire about the teams plan for my company."
Category: Sales

Email: "I am facing 500 code when I try to go to the settings page."
Category: Technical


Email: {email}

Category:"""

    response = client.chat.completions.create(
                   model="openai/gpt-oss-120b",
                   messages=[{"role": "user", "content": prompt}],
                   temperature=0.1,
                   max_tokens=100
            )
    return response.choices[0].message.content.strip()

def chain_of_thought_analysis(email):
    """
    Ask model to show its reasoning step by step. 
    Good For: Complex decision making, debugging, transparency in production.
    """
    prompt = f"""
    Analyze this customer email step by step

    Email: {email}

    Think through this step by step:
        1. What is the main issue or request?
        2. What category does this belongs to (Billing/Technical/Sales)?
        3. What is the urgency level (Low/Medium/High)?
        4. What sentiment does the customer express?
    (Note: You can only select one at max while choosing between the category and urgency level.)
    Provide you analysis:"""

    response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250,
            )
    return response.choices[0].message.content.strip()



# ==================== DEMO ALL TECHNIQUES ====================
def demo_all_techniques():
    """Run all techniques on sample emails to compare approaches."""
    
    test_email = sample_emails[1]  
    
    print("=" * 70)
    print("TEST EMAIL:")
    print(test_email)
    print("=" * 70)
    
    # # Zero-shot
    # print("\n1. ZERO-SHOT (No examples):")
    # result = zero_shot_classification(test_email)
    # print(f"Result: {result}")

    # Few-shot
    # print("\n2. FEW-SHOT (With examples):")
    # result = few_shot_classification(test_email)
    # print(f"Result: {result}")

    # Chain of Thought
    print("\n3. CHAIN OF THOUGHT (Step-by-step reasoning):")
    result = chain_of_thought_analysis(test_email)
    print(f"Result:\n{result}")

    # # Multi-step
    # print("\n4. MULTI-STEP (Sequential API calls):")
    # result = multi_step_classification(test_email)
    # print(f"Step 1 - Extracted: {result['extracted_info']}")
    # print(f"Step 2 - Category: {result['category']}")
    # print(f"Total API calls: {result['total_api_calls']}")


# ==================== PRODUCTION COMPARISON ====================
def production_comparison():
    """
    Compare techniques on production metrics:
    - Consistency
    - Latency
    - Token usage
    - Accuracy
    """
    print("\n" + "=" * 70)
    print("PRODUCTION CONSIDERATIONS:")
    print("=" * 70)
    
    considerations = """
    ZERO-SHOT:
    ✓ Fastest (1 API call, minimal tokens)
    ✓ Cheapest
    ✗ Less consistent on edge cases
    ✗ No control over output format
    → Best for: Simple, well-defined tasks
    
    FEW-SHOT:
    ✓ More consistent outputs
    ✓ Better format control
    ✓ Handles edge cases better
    ✗ Uses more input tokens (examples in every call)
    ✗ Still just 1 API call
    → Best for: Production classification, formatting needs
    
    CHAIN OF THOUGHT:
    ✓ Transparent reasoning
    ✓ Better accuracy on complex decisions
    ✓ Easier to debug
    ✗ Slower (more output tokens)
    ✗ Costs more
    → Best for: Complex analysis, debugging, explainability
    
    MULTI-STEP:
    ✓ Handles complex workflows
    ✓ Each step can use different models
    ✓ Can add validation between steps
    ✗ Slowest (multiple API calls)
    ✗ Most expensive
    ✗ More failure points
    → Best for: Complex pipelines, when steps depend on each other
    """
    
    print(considerations)


if __name__ == "__main__":
    # Make sure to set your GROQ_API_KEY environment variable
    # export GROQ_API_KEY='your-api-key-here'
    
    demo_all_techniques()
    # production_comparison()

