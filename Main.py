from  sympy import symbols, factor
from math import sqrt , sin , cos , tan , radians, factorial , isclose , perm , comb
from numpy import linalg , array , dot , cross 
import re
import numpy as np
from termcolor import colored
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, xticks, grid, axhline, axvline, xlim, show


class Math():
    try:
        
        def basic_math(self ):
            while True:
                expression = input(colored("\nEnter Expression: ","yellow"))
                if expression == '0' :
                    break
                try:
                    result = eval(expression)
                    print( result )
                except Exception as e:
                    print( f"Error: {str(e)}" )
        
        def chapter1(self):
                # Function to perform operations on complex numbers
                def perform_operation(operation):
                    operation = operation.replace("A",f"({complex_num1})").replace("B",f"({complex_num2})")
                    result = eval(operation)
                    return result
                # Get user inputs
                try:
                    while True:
                        num1 = input(colored("Enter the A complex number: ","yellow")).replace("i", "j")
                        num2 = input(colored("Enter the B complex number: ","yellow")).replace("i", "j")

                        if num1=='0' and num2=='0':
                            break
                        
                        # Convert inputs to complex numbers
                        complex_num1 = complex(num1)
                        complex_num2 = complex(num2)
                        
                        while True:
                            operation = input(colored(f"Enter the operation or type 'exit' to quit: ","cyan"))
                            result = perform_operation(operation)
                            print(colored(f"Result of {complex_num1} {operation} {complex_num2}: {result}".replace("j","i"),"green"))

                except ValueError as e:
                    print(colored(f"Error: {e}","red"))
                except Exception as e:
                    print(colored(f"An error occurred: {e}","red"))
        
        def chapter2(self):
            def matrices():
                # Function to get matrix input
                def get_matrix(order_input):
                    order_map = {
                        '1': (1, 1),
                        '2': (1, 2),
                        '3': (1, 3),
                        '4': (2, 1),
                        '5': (2, 2),
                        '6': (2, 3),
                        '7': (3, 1),
                        '8': (3, 2),
                        '9': (3, 3)
                    }
                    rows, cols = order_map[order_input]
                    matrix = []
                    print(f"Enter {rows * cols} elements for the matrix:")
                    for i in range(rows):
                        row = []
                        for j in range(cols):
                            element = input(f"Enter element ({i+1}, {j+1}): ")
                            row.append(float(element))  # Convert to float for numeric operations
                        matrix.append(row)
                    return matrix

                while True:
                    # Get first matrix
                    order1 = input(colored("""\nEnter the order of matrix
                                1.  1x1
                                2.  1x2
                                3.  1x3
                                4.  2x1
                                5.  2x2
                                6.  2x3
                                7.  3x1
                                8.  3x2
                                9.  3x3
                                        :""","yellow"))
                    if order1=='0':
                        break
                    matrix1 = get_matrix(order1)

                    # Get second matrix
                    order2 = input(colored("""\nEnter the order of matrix
                                1.  1x1
                                2.  1x2
                                3.  1x3
                                4.  2x1
                                5.  2x2
                                6.  2x3
                                7.  3x1
                                8.  3x2
                                9.  3x3
                                        :""","yellow"))
                    matrix2 = get_matrix(order2)

                    # Convert matrices to numpy arrays
                    A = array(matrix1)
                    B = array(matrix2)

                    try:
                        # Matrix addition
                        if A.shape == B.shape:
                            C = A + B
                            print(colored("\nMatrix A + B:" , "blue"))
                            print(C)
                        else:
                            print(colored("Error: Matrices must have the same dimensions for addition.","red" ))
                        
                        if A.shape == B.shape:
                            C = A - B
                            print(colored("\nMatrix A - B:" , "blue"))
                            print(C)
                        else:
                            print(colored("Error: Matrices must have the same dimensions for addition.","red" ))

                        # Matrix multiplication
                        if A.shape[1] == B.shape[0]:
                            D = dot(A, B)
                            print(colored("\nMatrix A * B (Multiplication):" , "cyan"))
                            print(D)
                        else:
                            print(colored("Error: Matrices cannot be multiplied due to incompatible dimensions." , "red"))
                            print(colored("Error: Matrices must have the same dimensions for addition.","red" ))

                        # Matrix multiplication
                        if A.shape[1] == B.shape[0]:
                            D = dot(B, A)
                            print(colored("\nMatrix B * A (Multiplication):" , "cyan"))
                            print(D)
                        else:
                            print(colored("Error: Matrices cannot be multiplied due to incompatible dimensions." , "red"))

                        # Determinant and inverse of matrix A
                        if A.shape[0] == A.shape[1]:
                            det_A = linalg.det(A)
                            inv_A = linalg.inv(A) if det_A != 0 else "Not invertible"
                            print(colored(f"\nDeterminant of A: {det_A:.1f}","blue"))
                            print(colored("\nInverse of A:","cyan"))
                            print(inv_A)
                        else:
                            print(colored("\nMatrix A is not square, determinant and inverse not possible.","red"))

                        # Determinant and inverse of matrix B
                        if B.shape[0] == B.shape[1]:
                            det_B = linalg.det(B)
                            inv_B = linalg.inv(B) if det_B != 0 else "Not invertible"
                            print(colored(f"\nDeterminant of B: {det_B:.2f}","blue"))
                            print(colored("\nInverse of B:","cyan"))
                            print(inv_B)
                        else:
                            print(colored("\nMatrix B is not square, determinant and inverse not possible.","red"))
                    
                    except Exception as e:
                        print(f"Error during matrix operations: {e}")
            matrices()
        
        def chapter3(self):
            def vectors():
                while True:
                    # Step 1: Take the number of vectors
                    num_vectors = int(input(colored("Enter the number of vectors: ","blue")))
                    if num_vectors=='0':
                        break

                    vectors = []
                    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']  # Extend if you need more labels
                    
                    # Step 2: Collect each vector's i, j, k components
                    for n in range(num_vectors):
                        label = labels[n]
                        print(colored(f"\nEnter components for vector {label}:","cyan"))
                        i = float(input(f"Enter i component for {label}: "))
                        j = float(input(f"Enter j component for {label}: "))
                        k = float(input(f"Enter k component for {label}: "))
                        vectors.append([i, j, k])
                    
                    vectors = array(vectors)
                    print(colored("\nVectors entered:","light_green"))
                    for idx, vector in enumerate(vectors):
                        print(f"{labels[idx]}: {vector}")
                    
                    # Step 3: Calculate and return unit vectors for each vector
                    unit_vectors = []
                    for vector in vectors:
                        magnitude = linalg.norm(vector)  # Calculate the magnitude
                        unit_vector = vector / magnitude if magnitude != 0 else [0, 0, 0]  # Handle zero vector case
                        unit_vectors.append(unit_vector)
                    
                    print(colored("\nUnit vectors:","blue"))
                    for idx, unit_vector in enumerate(unit_vectors):
                        rounded_vector = [f"{component:.1f}" for component in unit_vector]  # Format each component to 1 decimal place
                        print(colored(f"Unit vector of {labels[idx]}: {rounded_vector}","cyan"))
                    
                    # Step 4: Perform operations on vectors
                    while True:
                        operation = input(colored("\nEnter the operation (e.g., A+B, A-B etc or 'exit' to stop): ","cyan")).strip()
                        
                        if operation.lower() == 'exit':
                            break

                        try:
                            # Handle scalar multiplication (e.g., 2*A)
                            if '*' in operation:
                                scalar, vector_label = operation.split('*')
                                scalar = float(scalar.strip())
                                vector = vectors[labels.index(vector_label.strip())]
                                result = scalar * vector
                                print(colored(f"\nResult of {scalar} * {vector_label}: {result}","light_green"))

                            # Handle magnitude operation (e.g., |A+B| or |B|)
                            elif operation.startswith('|') and operation.endswith('|'):
                                inside = operation[1:-1]  # Extract the part inside the vertical bars
                                
                                if len(inside) == 1:  # Single vector magnitude (e.g., |A|)
                                    vector_label = inside
                                    vector = vectors[labels.index(vector_label)]
                                    magnitude = linalg.norm(vector)
                                    print(colored(f"\nMagnitude of {vector_label}: {magnitude:.2f}","light_green"))
                                elif len(inside) == 3:  # Operation inside magnitude (e.g., |A+B|)
                                    vector1_label, operator, vector2_label = inside[0], inside[1], inside[2]

                                    vector1 = vectors[labels.index(vector1_label)]
                                    vector2 = vectors[labels.index(vector2_label)]

                                    if operator == '+':
                                        result = vector1 + vector2
                                    elif operator == '-':
                                        result = vector1 - vector2
                                    else:
                                        raise ValueError(colored("Invalid operator in magnitude operation.","light_red"))
                                    
                                    magnitude = linalg.norm(result)
                                    print(colored(f"\nMagnitude of {vector1_label} {operator} {vector2_label}: {magnitude:.2f}","light_green"))
                                else:
                                    raise ValueError(colored("Invalid magnitude operation format.","light_red"))
                            else:
                                # Extract vectors from the input operation
                                vector1_label, operator, vector2_label = operation[0], operation[1], operation[2]

                                vector1 = vectors[labels.index(vector1_label)]
                                vector2 = vectors[labels.index(vector2_label)]

                                if operator == '+':
                                    result = vector1 + vector2
                                    print(colored(f"\nResult of {vector1_label} + {vector2_label}: {result}","light_cyan"))
                                elif operator == '-':
                                    result = vector1 - vector2
                                    print(colored(f"\nResult of {vector1_label} - {vector2_label}: {result}","light_cyan"))
                                elif operator == '.':  # Dot product
                                    result = dot(vector1, vector2)
                                    print(colored(f"\nDot product of {vector1_label} · {vector2_label}: {result}","light_cyan"))
                                elif operator == 'x':  # Cross product
                                    result = cross(vector1, vector2)
                                    print(colored(f"\nCross product of {vector1_label} × {vector2_label}: {result}","light_cyan"))
                                else:
                                    print(colored("Invalid operator. Please use +, -, ., × or '*' for valid operations."))
                        except (ValueError, IndexError):
                            print(colored("Invalid operation format or vector label. Please try again.","light_red"))
                    
                    return unit_vectors
            vectors()

        def chapter4(self):
            # Define formulas in a dictionary, each associated with the variables they compute
            formulas = {
                "an_arithmetic": {
                    "description": "Arithmetic sequence (nth term): an = a1 + (n - 1) d",
                    "variables": ["a1", "n", "d"],
                    "function": lambda a1, n, d: a1 + (n - 1) * d
                },
                "an_geometric": {
                    "description": "Geometric sequence (nth term): an = a1 * r^(n - 1)",
                    "variables": ["a1", "n", "r"],
                    "function": lambda a1, n, r: a1 * (r ** (n - 1))
                },
                "Sn_arithmetic": {
                    "description": "Sum of first n terms (Arithmetic): Sn = (n / 2) (2a1 + (n - 1) d)",
                    "variables": ["a1", "n", "d"],
                    "function": lambda a1, n, d: (n / 2) * (2 * a1 + (n - 1) * d)
                },
                "Sn_geometric": {
                    "description": "Sum of first n terms (Geometric): Sn = a1 (1 - r^n) / (1 - r)",
                    "variables": ["a1", "r", "n"],
                    "function": lambda a1, r, n: a1 * (1 - r ** n) / (1 - r) if r != 1 else a1 * n
                },
                "S_∞_geometric": {
                    "description": "Sum to ∞ (Geometric): S_∞ = a1 / (1 - r)",
                    "variables": ["a1", "r"],
                    "function": lambda a1, r: a1 / (1 - r) if abs(r) < 1 else None
                },
                "S_∞_alternate": {
                    "description": "Sum to ∞ (Alternate): S_∞ = (a / (1 - r)) + (d * r) / (1 - r)^2",
                    "variables": ["a", "r", "d"],
                    "function": lambda a, r, d: (a / (1 - r)) + (d * r / (1 - r) ** 2) if abs(r) < 1 else None
                },
                "mean_arithmetic": {
                    "description": "Arithmetic mean: A = (a + b) / 2",
                    "variables": ["a", "b"],
                    "function": lambda a, b: (a + b) / 2
                },
                "mean_geometric": {
                    "description": "Geometric mean: G = sqrt(a * b)",
                    "variables": ["a", "b"],
                    "function": lambda a, b: sqrt(a * b)
                },
                "mean_harmonic": {
                    "description": "Harmonic mean: H = 2ab / (a + b)",
                    "variables": ["a", "b"],
                    "function": lambda a, b: 2 * a * b / (a + b)
                },
                "Sn_alternate_1": {
                    "description": "Sum (Alternate): Sn = (a1 - an * r) / (1 - r)",
                    "variables": ["a1", "an", "r"],
                    "function": lambda a1, an, r: (a1 - (an * r)) / (1 - r) if r != 1 else None
                },
                "Sn_alternate_2": {
                    "description": "Sum (Another Alternate): Sn = (a / (1 - r)) + d * r (1 - r^n) / (1 - r)^2 - (a + nd) r^n / (1 - r)",
                    "variables": ["a", "r", "d", "n"],
                    "function": lambda a, r, d, n: (a / (1 - r)) + d * r * (1 - r ** n) / (1 - r) ** 2 - ((a + n * d) * r ** n / (1 - r)) if abs(r) < 1 else None
                },
                "Sn_alternate_3": {
                    "description": "Sum (Final Alternate): Sn = (a / (1 - r)) + d * r (1 - r^n) / (1 - r)^2",
                    "variables": ["a", "r", "d", "n"],
                    "function": lambda a, r, d, n: (a / (1 - r)) + (d * r * (1 - r ** n)) / (1 - r) ** 2 if abs(r) < 1 else None
                }
            }

            # Helper function to prompt user for variable values
            def get_variable_values(variable_names):
                values = {}
                for var in variable_names:
                    values[var] = float(input(f"Enter value for {var}: "))
                return values

            # Main function to choose formula and compute result
            def find_variable():
                while True:
                    # Step 1: Ask which variable the user wants to calculate
                    variable_to_find = input("Which variable do you want to find (e.g., an, Sn, S_∞, mean)? ").lower()
                    if variable_to_find=='0':
                        break
                    # Step 2: Show available formulas for that variable (case-insensitive)
                    relevant_formulas = {key: value for key, value in formulas.items() if variable_to_find in key.lower()}
                    
                    if not relevant_formulas:
                        print(f"No formulas available to find {variable_to_find}.")
                        return
                    
                    print(f"\nAvailable formulas to find {variable_to_find}:")
                    for idx, (key, formula) in enumerate(relevant_formulas.items(), start=1):
                        print(f"{idx}. {formula['description']}")
                    
                    # Step 3: Let user choose a formula
                    choice = int(input("\nChoose a formula by number: ")) - 1
                    selected_formula_key = list(relevant_formulas.keys())[choice]
                    selected_formula = relevant_formulas[selected_formula_key]
                    
                    # Step 4: Get the necessary variables
                    variable_names = selected_formula["variables"]
                    print(f"\nYou need to input the following variables: {', '.join(variable_names)}")
                    values = get_variable_values(variable_names)
                    
                    # Step 5: Compute and show the result
                    result = selected_formula["function"](**values)
                    print(f"\nThe result for {variable_to_find} is: {result}")

            find_variable()

        def chapter5(self):
            def factorize():
                while True:
                    equation= input(colored("\nEnter the equation:","yellow"))
                    if equation=='0':
                        break
                    # Remove spaces and handle '+' sign
                    equation = equation.replace(" ", "").replace("+", "")
                    
                    # Use regex to find coefficients for quadratic equations
                    pattern_quadratic = r"([+-]?\d*)x\^2([+-]?\d*)x([+-]?\d+)"
                    match_quadratic = re.match(pattern_quadratic, equation)

                    if match_quadratic:
                        # Extract coefficients for a quadratic equation
                        a = int(match_quadratic.group(1)) if match_quadratic.group(1) not in ["", "+", "-"] else int(match_quadratic.group(1) + "1")
                        b = int(match_quadratic.group(2)) if match_quadratic.group(2) not in ["", "+", "-"] else int(match_quadratic.group(2) + "1")
                        c = int(match_quadratic.group(3))

                        # Calculate the discriminant
                        discriminant = b**2 - 4 * a * c

                        if discriminant > 0:
                            # Two real and distinct roots
                            root1 = (-b + sqrt(discriminant)) / (2 * a)
                            root2 = (-b - sqrt(discriminant)) / (2 * a)
                            print(f"{a}(x - {root1})(x - {root2})")
                        elif discriminant == 0:
                            # One real root (repeated)
                            root = -b / (2 * a)
                            print(f"{a}(x - {root})^2")
                        else:
                            # Complex roots
                            print("The quadratic cannot be factorized into real numbers (complex roots).")
                        return
                    
                    # Check for cubic equations
                    pattern_cubic = r"([+-]?\d*)x\^3([+-]?\d*)x\^2([+-]?\d*)x([+-]?\d+)"
                    match_cubic = re.match(pattern_cubic, equation)

                    if match_cubic:
                        a = int(match_cubic.group(1)) if match_cubic.group(1) not in ["", "+", "-"] else int(match_cubic.group(1) + "1")
                        b = int(match_cubic.group(2)) if match_cubic.group(2) not in ["", "+", "-"] else int(match_cubic.group(2) + "1")
                        c = int(match_cubic.group(3)) if match_cubic.group(3) not in ["", "+", "-"] else int(match_cubic.group(3) + "1")
                        d = int(match_cubic.group(4))

                        # Define the variable
                        x = symbols("x")

                        # Define the cubic polynomial
                        polynomial = a * x**3 + b * x**2 + c * x + d

                    else:
                        # Check for quartic equations
                        pattern_quartic = r"([+-]?\d*)x\^4([+-]?\d*)x\^3([+-]?\d*)x\^2([+-]?\d*)x([+-]?\d+)"
                        match_quartic = re.match(pattern_quartic, equation)

                        if match_quartic:
                            a = int(match_quartic.group(1)) if match_quartic.group(1) not in ["", "+", "-"] else int(match_quartic.group(1) + "1")
                            b = int(match_quartic.group(2)) if match_quartic.group(2) not in ["", "+", "-"] else int(match_quartic.group(2) + "1")
                            c = int(match_quartic.group(3)) if match_quartic.group(3) not in ["", "+", "-"] else int(match_quartic.group(3) + "1")
                            d = int(match_quartic.group(4)) if match_quartic.group(4) not in ["", "+", "-"] else int(match_quartic.group(4) + "1")
                            e = int(match_quartic.group(5))

                            # Define the variable
                            x = symbols("x")

                            # Define the quartic polynomial
                            polynomial = a * x**4 + b * x**3 + c * x**2 + d * x + e
                        else:
                            print("Invalid equation format.")
                            return

                    # Factor the polynomial
                    factorized_form = factor(polynomial)
                    factorized_form = str(factorized_form)
                    factorized_form = factorized_form.replace("**", "^").replace("*", "")

                    print(f"Factorized form: {factorized_form}")
            factorize()
        
        def chapter6(self):
            def factorials():
                expression = input(colored("\nEnter the expression: ","yellow"))
                # Find all numbers followed by '!' and replace them with their factorial values
                def replace_factorial(match):
                    number = int(match.group(1))
                    return str(factorial(number))

                # Replace all occurrences of "number!" with the computed factorial
                expression_with_factorials = re.sub(r'(\d+)!', replace_factorial, expression)
                
                # Evaluate the final expression safely
                try:
                    result = eval(expression_with_factorials)
                    print( colored( result ,"light_green") )
                except Exception as e:
                    print( colored( f"Error evaluating expression: {e}","red" ) )
            
            def permutations(self):
                n= input(colored("\nn: ","yellow"))
                r= input(colored("\nr: ","yellow"))
                print(perm(int(n), int(r)))  # Using the built-in perm function
            
            def combinations(self):
                n= input(colored("\nn: ","yellow"))
                r= input(colored("\nr: ","yellow"))
                print(comb(int(n), int(r)))  # Using the built-in perm function
            
            while True:
                operation = input(colored("Enter\n1. Factorial\n2. Permutation\n3. Combination :","light_yellow"))
                if operation=='0':
                    break
                match operation:
                    case '1':
                        factorials()
                    case '2':
                        permutations()
                    case '3':
                        combinations()

        def chapter7(self):
            def binomial_expansion(a: str, b: str, n: str, a_val: float, b_val: float):
                # Convert n from string to integer
                n = int(n)
                
                expansion_terms = []
                term_results = []
                
                for r in range(n + 1):
                    coeff = comb(n, r)  # Calculate C(n, r)
                    
                    # Determine the term format based on the coefficient and term position
                    if r == 0:
                        term = f"({a}^{n})"  # First term
                    elif r == n:
                        term = f"({b}^{n})"  # Last term
                    else:
                        if coeff == 1:
                            term = f"({a}^{n - r})({b}^{r})"  # Coefficient of 1
                        else:
                            term = f"{coeff}({a}^{n - r})({b}^{r})"  # General term

                    expansion_terms.append(term)
                    
                    # Calculate the term value and store it
                    term_value = coeff * (a_val ** (n - r)) * (b_val ** r)
                    term_results.append((r + 1, coeff, a_val ** (n - r), b_val ** r, term_value))

                expansion = " + ".join(expansion_terms)
                
                # Evaluate the overall expression
                total_result = sum(term_results[i][4] for i in range(len(term_results)))

                return expansion, total_result, term_results

            # Example usage
            while True:
                a = input(colored("\n(a+b)^n\nEnter the value of a: ","light_blue"))
                b = input(colored("Enter the value of b: ","light_blue"))
                n = input(colored("Enter the value of n: ","light_blue"))
                if a=='0' and b=='0' and n=='0':
                    break
                a_val = int(a)  # Convert string to integer for a
                b_val = int(b)  # Convert string to integer for b

                expansion, evaluated_result, term_results = binomial_expansion(a, b, n, a_val, b_val)

                print(f"The expansion of ({a} + {b})^{n} is: {expansion}")
                print(f"The evaluated result is: {evaluated_result}\n")

                # Display each term and its value
                print("Terms and their values:")
                n=int(n)
                for r, coeff, a_term, b_term, term_value in term_results:
                    print(f"T_{r} = {coeff} * {a}^{n - (r - 1)} * {b}^{(r - 1)} = {term_value}")

        def trigonometric(self):
            while True:
                equation = input(colored("\nEnter\ntan(x) not  tan (x): ","yellow"))

                if equation=='0':
                    break
                
                # Dictionary to map trigonometric function names to actual math functions
                trig_functions = {
                    "sin": sin,
                    "cos": cos,
                    "tan": tan,
                    "cosec": lambda x: 1 / sin(x) if sin(x) != 0 else float('∞'),
                    "sec": lambda x: 1 / cos(x) if cos(x) != 0 else float('∞'),
                    "cot": lambda x: 1 / tan(x) if tan(x) != 0 else float('∞'),
                }

                # Function to evaluate half-angle formulas
                def evaluate_half_angle(trig_name, angle_radians):
                    if trig_name == "sin":
                        return sqrt((1 - cos(angle_radians)) / 2)
                    elif trig_name == "cos":
                        return sqrt((1 + cos(angle_radians)) / 2)
                    elif trig_name == "tan":
                        return sqrt((1 - cos(angle_radians)) / (1 + cos(angle_radians)))

                # Function to evaluate double-angle formulas
                def evaluate_double_angle(trig_name, angle_radians):
                    if trig_name == "sin":
                        return 2 * sin(angle_radians) * cos(angle_radians)
                    elif trig_name == "cos":
                        return cos(angle_radians) ** 2 - sin(angle_radians) ** 2
                    elif trig_name == "tan":
                        return 2 * tan(angle_radians) / (1 - tan(angle_radians) ** 2)

                # Function to evaluate triple-angle formulas
                def evaluate_triple_angle(trig_name, angle_radians):
                    if trig_name == "sin":
                        return 3 * sin(angle_radians) - 4 * sin(angle_radians) ** 3
                    elif trig_name == "cos":
                        return 4 * cos(angle_radians) ** 3 - 3 * cos(angle_radians)
                    elif trig_name == "tan":
                        t = tan(angle_radians)
                        return (3 * t - t ** 3) / (1 - 3 * t ** 2) if 1 - 3 * t ** 2 != 0 else float('∞')

                # Helper function to evaluate a single term
                def evaluate_single_term(term, trig_functions):
                    for trig_name in trig_functions:
                        if trig_name in term:
                            # Handle angles in parentheses
                            angle_str = term[term.index("(") + 1:term.index(")")]
                            
                            # Evaluate angle expression (e.g., "2*30")
                            try:
                                angle = eval(angle_str)  # Calculate the angle
                                radian_angle = radians(angle)  # Convert to radians
                                
                                # Handle special case for tan(90) and tan(270)
                                if trig_name == "tan" and (angle % 180) == 90:
                                    return float('∞')  # Representing as ∞
                            except Exception as e:
                                raise ValueError(f"Error evaluating angle '{angle_str}': {e}")

                            # Check for triple angle (e.g., sin(3*45))
                            if "3*" in term:  # Triple angle
                                return evaluate_triple_angle(trig_name, radian_angle)
                            # Check for double angle (e.g., sin(2*45))
                            elif "2*" in term:  
                                return evaluate_double_angle(trig_name, radian_angle)
                            # Check for half angle (e.g., sin(45/2))
                            elif "/2" in term:  
                                return evaluate_half_angle(trig_name, radian_angle)
                            else:  # Regular angle (e.g., sin(45))
                                return trig_functions[trig_name](radian_angle)

                    # If it's a plain number or an invalid function
                    try:
                        return float(term)  # Convert string to float if possible
                    except ValueError:
                        raise ValueError(f"Invalid term: '{term}'. Could not convert to float.")

                # Function to handle both simple and multiple expressions
                def handle_expressions(expr):
                    # Check for invalid function names
                    invalid_functions = ["cas", "inv"]  # Add more as necessary
                    for func in invalid_functions:
                        if func in expr:
                            raise ValueError(f"Invalid function: '{func}' found in expression.")

                    # Check if there is an equal sign (=) in the expression
                    if "=" in expr:
                        parts = expr.split("=")
                        values = []

                        for part in parts:
                            value = evaluate_expression(part.strip())
                            values.append(value)
                            print(f"Evaluated: {part.strip()} = {value}")

                        # Check if all evaluated values are equal
                        all_equal = all(isclose(values[i], values[i + 1], rel_tol=1e-9) for i in range(len(values) - 1))
                        if all_equal:
                            print("All sides are equal.")
                        else:
                            print("Sides are not equal.")
                    else:
                        value = evaluate_expression(expr.strip())
                        print(f"Evaluated expression: {value:.4f}")

                # Function to evaluate a single expression
                def evaluate_expression(expr):
                    terms = expr.replace(" ", "").split("+")  # Split on plus signs
                    total = 0  # Initialize total

                    for term in terms:
                        if "-" in term:
                            sub_terms = term.split("-")
                            total += evaluate_single_term(sub_terms[0], trig_functions)
                            total -= sum(evaluate_single_term(st, trig_functions) for st in sub_terms[1:])
                        else:
                            total += evaluate_single_term(term, trig_functions)

                    return total

                handle_expressions(equation)  # Process the expression

        def graph(self):
            while True:
                # Prompt for input
                variable = input(colored("Enter variable\ntan(2x) etc (enter '0' to exit): ", "light_magenta"))
                if variable == '0':
                    break

                # Parse the variable input
                value = variable.split("(")[0]
                angle = variable.split("(")[-1].split("x)")[0]
                angle = int(angle) if angle else 1

                # Define x values (positive and negative)
                x_values = np.array([0,  10,  20,  30,  40,  50,  60,  70,  80, 90, 
                                    100, 110, 120, -120, -110, -100, 
                                    -90, -80, -70, -60, -50, -40, -30, -20, -10, 0])

                # Calculate sine, cosine, and tangent values
                sin_x = np.sin(np.radians(angle * x_values))
                cos_x = np.cos(np.radians(angle * x_values))
                tan_x = np.tan(np.radians(angle * x_values))

                # Avoid division by zero using NumPy's where function
                cosec_x = np.where(sin_x != 0, 1 / sin_x, np.inf)
                sec_x = np.where(cos_x != 0, 1 / cos_x, np.inf)
                cot_x = np.where(tan_x != 0, 1 / tan_x, np.inf)

                # Select y_values based on the variable input
                if value == 'tan':
                    y_values = tan_x
                elif value == 'cos':
                    y_values = cosec_x
                elif value == 'sin':
                    y_values = sin_x
                elif value == 'cot':
                    y_values = cot_x
                elif value == 'cosec':
                    y_values = cosec_x
                elif value == 'sec':
                    y_values = sec_x
                else:
                    print("Unknown function. Please enter 'tan', 'sin', 'cos', 'cot', 'cosec', or 'sec'.")
                    continue  # Skip plotting if the function is unknown

                # Plot the graph
                figure(figsize=(10, 5))
                plot(x_values, y_values, marker='o', linestyle='-', color='b')
                title(f'Graph of {variable}')
                xlabel('x (degrees)')
                ylabel(f'{variable}')
                xticks(np.arange(-120, 130, 10))

                grid()
                axhline(0, color='black', linewidth=0.5, ls='--')
                axvline(0, color='black', linewidth=0.5, ls='--')
                xlim(-120, 120)
                show()
    
    except Exception as e:
        print(colored(f"Error:{Exception}", "red"))









# Create an instance of Math class
a = Math()

# Main loop to handle user input
while True:
    chapter = input(
        colored(
            """\nWhich Chapter you wanna solve? 
    0 to exit.
    1. Complex Numbers
    2. Matrices and Determinants
    3. Vectors
    4. Sequences and Series
    5. Polynomials
    6. Permutation and Combination
    7. Mathematical induction and Binomial Theorem
    8. Trignometry
    9. Graph :""",
            "light_magenta",
        )
    )

    match chapter:
        case "0":
            print("Exiting...")
            break
        case "1":
            a.chapter1()
        case "2":
            a.chapter2()
        case "3":
            a.chapter3()
        case "4":
            a.chapter4()
        case "5":
            a.chapter5()
        case "6":
            a.chapter6()
        case "7":
            a.chapter7()
        case "8":
            a.trigonometric()
        case "9":
            a.graph()
        case _:
            print(colored("Invalid option, please try again.","red"))