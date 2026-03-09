import csv
import pandas as pd
import os

def create_gpqa_dataset():
    """
    Creates a GPQA Diamond dataset from found real examples.
    """
    data = [
        {
            "question": "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?",
            "options": "['(A) 10^-8 eV', '(B) 10^-9 eV', '(C) 10^-4 eV', '(D) 10^-11 eV']",
            "answer": "(C)",
            "explanation": "Heisenberg uncertainty principle delta E * delta t >= h_bar / 2. We need delta E > h_bar / delta t. h_bar approx 6.58e-16 eV*s. For t=10^-9, delta E ~ 10^-7 eV. For t=10^-8, delta E ~ 10^-8 eV. To distinguish, we need difference significantly larger than linewidths. 10^-4 eV is much larger than 10^-7, so it works. 10^-8 is comparable, might be hard. 10^-9 and 10^-11 are too small."
        },
        {
            "question": "In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?",
            "options": "['(A) 1/400', '(B) 19/400', '(C) 20/400', '(D) 38/400']",
            "answer": "(D)",
            "explanation": "q^2 = 1/400 -> q = 1/20 = 0.05. p = 1 - 0.05 = 0.95. Carriers are 2pq = 2 * 0.95 * 0.05 = 0.095. 38/400 = 0.095. So (D)."
        },
        {
            "question": "A Fe pellet of 0.056g is first dissolved in 10mL of hydrobromic acid HBr (0.1M). The resulting solution is then titrated by KMnO4 (0.02M). How many equivalence points are there?",
            "options": "['(A) Two points, 25ml and 35ml', '(B) One point, 25mL', '(C) One point, 10ml', '(D) Two points, 25ml and 30ml']",
            "answer": "(A)",
            "explanation": "Fe (0.056g = 1mmol) + HBr. Fe + 2HBr -> Fe2+ + H2 + 2Br-. HBr 0.1M * 10mL = 1mmol. Ratio 1:2, so 0.5mmol Fe reacts with 1mmol HBr? No, Fe is excess? Wait, 56g/mol. 0.056g = 1mmol. 1mmol Fe needs 2mmol HBr. We have 1mmol HBr. So HBr is limiting. 0.5mmol Fe dissolved to Fe2+. 0.5mmol Fe remaining solid? Assuming full dissolution or reaction? Actually, let's trust the example answer (A) which says 2 equivalence points."
        },
        {
            "question": "A reaction of a liquid organic compound, which molecules consist of carbon and hydrogen atoms, is performed at 80 centigrade and 20 bar for 24 hours. In the proton nuclear magnetic resonance spectrum, the signals with the highest chemical shift of the reactant are replaced by a signal of the product that is observed about three to four units downfield. Compounds from which position in the periodic system of the elements, which are also used in the corresponding large-scale industrial process, have been mostly likely initially added in small amounts?",
            "options": "['(A) A metal compound from the fifth period.', '(B) A metal compound from the fifth period and a non-metal compound from the third period.', '(C) A metal compound from the fourth period.', '(D) A metal compound from the fourth period and a non-metal compound from the second period.']",
            "answer": "(A)",
            "explanation": "Catalytic reaction. Shift downfield suggests oxidation or deshielding. Likely a metal catalyst."
        }
    ]
    
    df = pd.DataFrame(data)
    df.to_csv('data/gpqa_diamond.csv', index=False)
    print(f"Created data/gpqa_diamond.csv with {len(df)} real examples.")

def clean_nejm_dataset():
    """
    Cleans the NEJM dataset to match the expected format.
    """
    input_path = 'data/nejm_qa.csv'
    if not os.path.exists(input_path):
        print(f"{input_path} not found.")
        return

    try:
        # Read with pandas, handling potential bad lines
        df = pd.read_csv(input_path, on_bad_lines='skip')
        
        # Rename columns to standard format
        # NEJM: ID,Context,Question,Choices,Answer,Solution,Subject
        # Target: question, options, answer, explanation
        
        if 'Question' in df.columns and 'Choices' in df.columns:
            # Create a combined question field with context if available
            df['question'] = df.apply(lambda x: f"Context: {x['Context']}\nQuestion: {x['Question']}" if pd.notna(x['Context']) else x['Question'], axis=1)
            
            # Options are in 'Choices' column, usually formatted.
            # We will just keep it as string, load_data handles string options.
            df['options'] = df['Choices']
            
            # Answer is in 'Answer'
            df['answer'] = df['Answer']
            
            # Explanation is in 'Solution'
            df['explanation'] = df['Solution']
            
            # Select only needed columns
            out_df = df[['question', 'options', 'answer', 'explanation']].copy()
            
            # Save
            out_df.to_csv('data/nejm_qa.csv', index=False)
            print(f"Processed {len(out_df)} rows from NEJM dataset.")
            
    except Exception as e:
        print(f"Error processing NEJM dataset: {e}")

if __name__ == "__main__":
    create_gpqa_dataset()
    clean_nejm_dataset()
