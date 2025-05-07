import pandas as pd
import numpy as np
import os

def generate_preference_scores():
    # Read the CSV file
    df = pd.read_csv('employees.csv')
    
    # Create a list to store the results
    results = []
    
    # Process each employee
    for _, row in df.iterrows():
        employee_name = f"{row['firstName']} {row['lastName']}"
        
        # Dictionary to store scores for each plan
        emp_scores = {}
        
        # Define all plan types to process
        plan_types = [
            'plan_level_id1', 'plan_level_id2', 'plan_level_id3', 
            'plan_level_id4', 'plan_level_id5', 'hsa_eligible_plan',
            'reco_plan1', 'reco_plan3', 'reco_plan4'
        ]
        
        # Process each plan type
        for plan_type in plan_types:
            plan_name = row[plan_type]
            if pd.notna(plan_name):
                plan_name = str(plan_name)
                if plan_name not in emp_scores:
                    emp_scores[plan_name] = 0
                
                # Assign scores based on plan type
                if plan_type == 'hsa_eligible_plan':
                    emp_scores[plan_name] += 10
                elif plan_type == 'reco_plan1':
                    emp_scores[plan_name] += 8
                elif plan_type == 'reco_plan4':
                    emp_scores[plan_name] += 7
                elif plan_type == 'reco_plan3':
                    emp_scores[plan_name] += 6
                else:
                    emp_scores[plan_name] += 5
        
        # Normalize scores for this employee
        if emp_scores:
            min_score = min(emp_scores.values())
            max_score = max(emp_scores.values())
            if max_score > min_score:
                normalized_scores = {
                    plan: (score - min_score) / (max_score - min_score)
                    for plan, score in emp_scores.items()
                }
            else:
                normalized_scores = {plan: 1.0 for plan in emp_scores}
            
            # Add normalized results to the list
            for plan_name, score in normalized_scores.items():
                results.append({
                    'Employee Name': employee_name,
                    'Plan ID': plan_name,
                    'Preference Score': score
                })
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    
    # Sort by Employee Name and Preference Score (descending)
    result_df = result_df.sort_values(['Employee Name', 'Preference Score'], ascending=[True, False])
    
    # Save to CSV in current directory
    output_path = 'preference_scores.csv'
    result_df.to_csv(output_path, index=False)
    print(f"Preference scores have been generated and saved to '{output_path}'")
    
    # Display first few rows
    print("\nFirst few rows of the generated preference scores:")
    print(result_df.head())

if __name__ == "__main__":
    generate_preference_scores() 