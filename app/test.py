import json
import re
import sys
import uuid
from typing import List, Dict, Tuple
from core.rag import AdvancedRAG


# Class to represent test cases
class TestCase:
    def __init__(self, question: str, expected_answer: str, description: str = None):
        self.question = question
        self.expected_answer = expected_answer
        self.description = description or question

    def __str__(self):
        return f"Q: {self.question}\nExpected: {self.expected_answer}"

# Function to run the QA system and get its answer
def get_answer(question: str) -> str:
    """Run the RAG system and return its answer."""
    rag_system = AdvancedRAG(data_path="../data/stock_news.json")
    try:
        # Generate a unique conversation ID for each question
        conversation_id = str(uuid.uuid4())
        response = rag_system.process_query(question, conversation_id)
        # print(response)
        
        # Check if response is a dict with a 'content' field (adjust based on your system's response format)
        if isinstance(response, dict) and 'content' in response:
            return response['content']
        else:
            # Try to convert to string if not a dict with content
            return str(response)
    except Exception as e:
        print(f"Error running RAG system: {e}")
        return ""

# Function to check if the expected answer is contained in the actual answer
def check_answer(actual_answer: str, expected_answer: str) -> bool:
    """Check if the expected answer is contained in the actual answer."""
    # For numerical answers, extract numbers and compare
    if re.match(r'^\d+(\.\d+)?$', expected_answer):
        # Extract all numbers from the actual answer
        numbers = re.findall(r'\d+(?:\.\d+)?', actual_answer)
        return any(number == expected_answer for number in numbers)
    
    # For text answers, check if the expected answer is contained in the actual answer
    return expected_answer.lower() in actual_answer.lower()

# Define the test cases
def get_test_cases() -> List[TestCase]:
    return [
        TestCase(
            "How many paid memberships does Netflix have?",
            "302 million",
            "Netflix subscriber count"
        ),
        TestCase(
            "What was Netflix's revenue in 2024?",
            "39 billion",
            "Netflix revenue"
        ),
        TestCase(
            "What is Nvidia's position in AI accelerators market share?",
            "84%",
            "Nvidia market share"
        ),
        TestCase(
            "Who is the CEO of Apple?",
            "Tim Cook",
            "Apple CEO"
        ),
        TestCase(
            "What was the fourth quarter revenue for Amazon?",
            "$28.79 billion",
            "Amazon Q4 revenue"
        ),
        TestCase(
            "What is the growth rate of Illumina's earnings?",
            "13.4%",
            "Illumina earnings growth rate"
        ),
        TestCase(
            "How much did Intel's fourth quarter sales fall?",
            "7%",
            "Intel sales decline"
        ),
        TestCase(
            "What company is developing the BioNeMo platform?",
            "NVIDIA",
            "BioNeMo developer"
        ),
        TestCase(
            "What is Meta's AI model called?",
            "Llama",
            "Meta AI model name"
        ),
        TestCase(
            "What did GE and AWS unveil last year as part of their partnership?",
            "X-ray and MRI models",
            "Netflix viewing time increase on Facebook"
        ),
        TestCase(
            "What is IBM's partnership with Penn State about?",
            "MyResource",
            "IBM-Penn State partnership"
        ),
        TestCase(
            "What was Apple's operating margin?",
            "32%",
            "Apple operating margin"
        ),
        TestCase(
            "What is the expected CAGR for the live streaming market?",
            "16.6%",
            "Live streaming market CAGR"
        ),
        TestCase(
            "What AI chip did AMD create to compete with Nvidia?",
            "MI300",
            "AMD AI chip"
        ),
        TestCase(
            "What was Microsoft's revenue in Q2 2025?",
            "69.6 billion",
            "Microsoft Q2 2025 revenue"
        ),
        TestCase(
            "Which lager surpassed Bud Light in 2023 as the top selling beer in the U.S.?",
            "Modelo Especial",
            "Beating Bud Light"
        ),
        TestCase(
            "What percentage of the Fortune 500 is served by CyberArk?",
            "50%",
            "CyberArk Fortune 500 percentage"
        ),
        TestCase(
            "How much did CyberArk's fourth-quarter revenues jump year over year?",
            "40.9%",
            "CyberArk revenue"
        ),
        TestCase(
            "How much did Warren Buffett reduce his Apple holdings by?",
            "100 million shares",
            "Buffett Apple reduction"
        ),
        TestCase(
            "When has Walmart introduced Express Delivery?",
            "April 2021",
            "Walmart Express Delivery"
        ),
        TestCase(
            "What percentage of global active devices is Apple's installed base?",
            "2.35 billion",
            "Apple installed base"
        ),
        TestCase(
            "What was the global clinical data analytics market valued at in 2023?",
            "$15.5 billion",
            "global clinical data analytics market"
        ),
        TestCase(
            "What expected total free cash flow did AMC Networks provide for 2024-2025?",
            "550 million",
            "AMC Networks free cash flow guidance"
        ),
        TestCase(
            "What is the projected CAGR (Compound Annual Growth Rate) for the smart fleet management market from 2025 to 2034?",
            "11.55%",
            "CAGR smart fleet management market"
        ),
        TestCase(
            "What was the most watched Netflix original series?",
            "Squid Game",
            "Most popular Netflix show"
        ),
    ]

def run_tests(test_cases: List[TestCase]) -> Tuple[List[Dict], float]:
    """Run the tests and return results and accuracy."""
    results = []
    correct_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Running test {i}/{len(test_cases)}: {test_case.description}")
        
        # Get answer from RAG system
        actual_answer = get_answer(test_case.question)
        
        # Check if the answer is correct
        is_correct = check_answer(actual_answer, test_case.expected_answer)
        if is_correct:
            correct_count += 1
            
        # Store result
        results.append({
            "question": test_case.question,
            "expected_answer": test_case.expected_answer,
            "actual_answer": actual_answer,
            "is_correct": is_correct
        })
        
        # Print result
        status = "✓" if is_correct else "✗"
        print(f"{status} Expected: {test_case.expected_answer}")
        if not is_correct:
            print(f"  Actual: {actual_answer[:100]}..." if len(actual_answer) > 100 else f"  Actual: {actual_answer}")
        print()
    
    # Calculate accuracy
    accuracy = correct_count / len(test_cases) if test_cases else 0
    
    return results, accuracy

def main():
    # Get test cases
    test_cases = get_test_cases()
    
    # Run tests
    results, accuracy = run_tests(test_cases)
    
    # Print summary
    print(f"\nResults: {sum(r['is_correct'] for r in results)}/{len(results)} correct")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Save results to file
    with open("qa_test_results.json", "w") as f:
        json.dump({
            "results": results,
            "accuracy": accuracy
        }, f, indent=2)
    
    print(f"\nResults saved to qa_test_results.json")

if __name__ == "__main__":
    main()
