import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

# Load data
employees_df = pd.read_csv('employees.csv')
preferences_df = pd.read_csv('preference_scores.csv')

# Preprocess employees
employees = employees_df[['employeeId', 'firstName', 'lastName', 'composition']].copy()
employees['employeeId'] = employees['employeeId'].astype(str)

# Map full names to employee IDs for easier matching
name_to_id = {
    f"{row['firstName']} {row['lastName']}": row['employeeId']
    for _, row in employees.iterrows()
}

# Extract plans and premiums per employee
plan_columns = ['plan_level_id1', 'plan_level_id2', 'plan_level_id3', 'plan_level_id4', 'plan_level_id5', 'hsa_eligible_plan']
premium_columns = [col + '_premium' for col in plan_columns]

# Create dictionary of available plans and their premiums for each employee
employee_plans = []
for _, row in employees_df.iterrows():
    emp_plans = {}
    for plan_col, premium_col in zip(plan_columns, premium_columns):
        plan_id = row[plan_col]
        if pd.notna(plan_id):
            premium = row[premium_col]
            emp_plans[str(plan_id)] = float(premium)
    employee_plans.append(emp_plans)

# Build all unique plans
all_plans = set()
for ep in employee_plans:
    all_plans.update(ep.keys())
all_plans = list(all_plans)
plan_indices = {plan: idx for idx, plan in enumerate(all_plans)}
num_employees = len(employees_df)
num_plans = len(all_plans)

# Preference dictionary
pref_dict = preferences_df.set_index(['Employee Name', 'Plan ID'])['Preference Score'].to_dict()

# Decision variables
# Create variables for each employee-plan combination
x = {}
for i in range(num_employees):
    for plan in employee_plans[i]:
        x[(i, plan)] = LpVariable(f"x_{i}_{plan}", cat="Binary")

# Contribution factors (as constants)
c1 = 0.8  # Employee Only
c2 = 0.7  # Employee + Spouse
c3 = 0.6  # Employee + Family

# Objective function: minimize total company cost
problem = LpProblem("Health_Plan_Assignment", LpMinimize)

# Total cost expression
total_cost_expr = lpSum(
    employee_plans[i][plan] * x[(i, plan)] * (
        c1 if employees.iloc[i]['composition'] == 'Employee Only' else
        c2 if employees.iloc[i]['composition'] == 'Employee + Spouse' else
        c3
    )
    for i in range(num_employees)
    for plan in employee_plans[i]
)

# Preference score expression
total_preference_expr = lpSum(
    pref_dict.get((f"{employees.iloc[i]['firstName']} {employees.iloc[i]['lastName']}", plan), 0) * x[(i, plan)]
    for i in range(num_employees)
    for plan in employee_plans[i]
)

# Combine into single objective with weights
weight_cost = 1.0
weight_pref = -0.001  # Adjust this to balance between cost and preference

problem += weight_cost * total_cost_expr + weight_pref * total_preference_expr, "MultiObjective"

# Constraints

# 1. Each employee must be assigned exactly one plan
for i in range(num_employees):
    problem += lpSum(
        x[(i, plan)]
        for plan in employee_plans[i]
    ) == 1, f"One_Plan_Employee_{i}"

# 2. Employees can only be assigned available plans
for i in range(num_employees):
    for plan in all_plans:
        if plan not in employee_plans[i]:
            if (i, plan) in x:
                problem += x[(i, plan)] == 0, f"No_Plan_{i}_{plan}"

# Solve the problem
print("Solving optimization...")
status = problem.solve()
print(f"Status: {status}")

# Output results
assignments = []
total_company_cost = 0
for i in range(num_employees):
    emp_name = f"{employees.iloc[i]['firstName']} {employees.iloc[i]['lastName']}"
    comp = employees.iloc[i]['composition']
    
    for plan in employee_plans[i]:
        if value(x[(i, plan)]) > 0.5:
            contrib_factor = (
                c1 if comp == 'Employee Only' else
                c2 if comp == 'Employee + Spouse' else
                c3
            )
            premium = employee_plans[i][plan]
            total_company_cost += premium * contrib_factor
            assignments.append({
                'Employee Name': emp_name,
                'Composition': comp,
                'PlanID': plan,
                'Contribution Factor': round(contrib_factor, 2),
                'Premium': premium
            })
            break

# Group by composition to get average contribution
contrib_summary = pd.DataFrame(assignments)[['Composition', 'Contribution Factor']]
contrib_summary = contrib_summary.groupby('Composition').mean().reset_index()

# Final output
print("\nEmployer Contribution in %")
print("----------------------------")
print(contrib_summary.to_string(index=False))

print("\n\nEmployee Assignments")
print("--------------------")
assign_df = pd.DataFrame(assignments)
print(assign_df[['Employee Name', 'PlanID', 'Contribution Factor', 'Premium']].to_string(index=False))

print("\n\nTotal Cost for Company")
print("----------------------")
print(f"${total_company_cost:.2f}")