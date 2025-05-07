import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpBinary, LpContinuous

# Load data
employees_df = pd.read_csv('employees.csv')
prefs_df = pd.read_csv('preference_scores.csv')

# Extract all available plans
plan_columns = ['plan_level_id1', 'plan_level_id2', 'plan_level_id3',
                'plan_level_id4', 'plan_level_id5', 'hsa_eligible_plan']
premium_columns = [col + '_premium' for col in plan_columns]

all_plans = set()
for col in plan_columns:
    all_plans.update(employees_df[col].dropna().unique())
all_plans = list(all_plans)

# Coverage types
coverage_types = ["Employee Only", "Employee + Spouse", "Employee + Family"]

# Map employees to their available plans, premiums, and preference scores
employee_data = []

for _, row in employees_df.iterrows():
    name = f"{row['firstName']} {row['lastName']}"
    coverage = row['composition']
    emp_plans = {}
    emp_prefs = {}

    # Get all plans and premiums
    for plan_col, premium_col in zip(plan_columns, premium_columns):
        plan_id = row[plan_col]
        if pd.isna(plan_id):
            continue
        premium = row[premium_col]
        if pd.notna(premium):
            emp_plans[plan_id] = float(premium)

    # Get preference scores from preference_scores.csv
    pref_rows = prefs_df[prefs_df['Employee Name'] == name]
    for _, p_row in pref_rows.iterrows():
        plan_id = p_row['Plan ID']
        score = p_row['Preference Score']
        emp_prefs[plan_id] = score

    employee_data.append({
        'name': name,
        'coverage': coverage,
        'plans': emp_plans,
        'prefs': emp_prefs
    })

# Create LP Problem
prob = LpProblem("Health_Plan_Selection", LpMinimize)

# Decision Variables
num_employees = len(employee_data)
x = [[LpVariable(f"x_{i}_{j}", cat=LpBinary) for j in range(len(all_plans))] for i in range(num_employees)]

# y[j]: Whether plan j is used
y = [LpVariable(f"y_{j}", cat=LpBinary) for j in range(len(all_plans))]

# Contribution factors per coverage type
contrib_vars = {
    ct: LpVariable(f"contrib_{ct.replace(' ', '_')}", lowBound=0.1, upBound=1.0, cat=LpContinuous)
    for ct in coverage_types
}

# Calculate normalization factors
max_cost = sum(sum(emp['plans'].values()) for emp in employee_data)
max_pref = sum(sum(emp['prefs'].values()) for emp in employee_data if emp['prefs'])
norm_factor = max(max_cost, max_pref) if max_cost and max_pref else 1

print(f"Normalization factor: {norm_factor}")

# Auxiliary variable for cost: z[i][j] = contrib * premium * x[i][j]
z = {}
cost_terms = []
pref_terms = []

for i in range(num_employees):
    emp = employee_data[i]
    coverage = emp['coverage']
    contrib = contrib_vars[coverage]

    for j, plan in enumerate(all_plans):
        if plan not in emp['plans']:
            prob += x[i][j] == 0
            continue

        premium = emp['plans'][plan]
        pref = emp['prefs'].get(plan, 0)

        # Create auxiliary variable for cost
        z[(i, j)] = LpVariable(f"z_{i}_{j}", lowBound=0)

        # Enforce z == contrib * premium * x[i][j] using Big-M method
        M = premium  # Big-M value based on max possible premium
        prob += z[(i, j)] <= premium * contrib, f"z_ub1_{i}_{j}"
        prob += z[(i, j)] <= M * x[i][j], f"z_ub2_{i}_{j}"
        prob += z[(i, j)] >= premium * contrib - M * (1 - x[i][j]), f"z_lb_{i}_{j}"
        prob += z[(i, j)] >= 0, f"z_pos_{i}_{j}"

        # Add terms to objective
        cost_terms.append(z[(i, j)])
        pref_terms.append(pref * x[i][j])

# Objective: minimize cost - preference
alpha = 0.5  # weight on cost
beta = 0.5   # weight on preference
prob += (alpha * lpSum(cost_terms) / norm_factor) - (beta * lpSum(pref_terms) / norm_factor)

# Constraints

# Each employee gets exactly one plan
for i in range(num_employees):
    prob += lpSum(x[i]) == 1, f"one_plan_{i}"

# Link y[j] with x[i][j]
for j in range(len(all_plans)):
    for i in range(num_employees):
        prob += x[i][j] <= y[j], f"link_{i}_{j}"

# Max 5 unique plans
prob += lpSum(y) <= 5, "max_plans"

# At least one HSA-eligible plan must be selected
hsa_plans = employees_df['hsa_eligible_plan'].dropna().unique()
if len(hsa_plans) > 0:
    hsa_plan_indices = [all_plans.index(p) for p in hsa_plans if p in all_plans]
    if hsa_plan_indices:
        prob += lpSum(y[j] for j in hsa_plan_indices) >= 1, "hsa_requirement"

# Solve model
print("\nSolving optimization problem...")
status = prob.solve()

if status != 1:
    print("❌ Problem is infeasible or solver failed.")
else:
    print("\n✅ Solution found!")

    # Output Results
    contrib_output = {ct: round(value(contrib_vars[ct]) * 100, 1) for ct in coverage_types}

    print("\nEmployer Contribution Percentages:")
    for ct in coverage_types:
        print(f"{ct:<20} : {contrib_output[ct]}%")

    print("\nEmployee Assignments:")
    print("Employee Name         Plan ID           Premium       Contribution")
    print("-" * 70)

    total_cost = 0
    for i in range(num_employees):
        emp = employee_data[i]
        for j in range(len(all_plans)):
            if value(x[i][j]) > 0.5:
                plan = all_plans[j]
                coverage = emp['coverage']
                contrib_percent = contrib_output[coverage]
                premium = emp['plans'][plan]
                cost = premium * contrib_percent / 100
                total_cost += cost
                print(f"{emp['name']:<20} {plan:<15} ${premium:>8.2f}     {contrib_percent:>5.1f}%")

    print(f"\nTotal Cost for Company: ${total_cost:.2f}")

    print("\nSelected Plans:")
    for j in range(len(all_plans)):
        if value(y[j]) > 0.5:
            print(f"- {all_plans[j]}")