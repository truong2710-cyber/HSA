import re

def extract(file_path):
    # file_path = './best_maximum_coverage_ratio.log'

    with open(file_path, 'r') as file:
        content = file.read()

    # Display the first 500 characters to understand its structure
    content[:500]
    # Extract all "Best harmony" and "Type" entries from the log file content
    best_harmony_pattern = re.compile(r"Best harmony: (\[\[.*?\]\])")
    type_pattern = re.compile(r"Type: (.+)")

    best_harmony_matches = best_harmony_pattern.findall(content)
    type_matches = type_pattern.findall(content)

    # Convert the Best harmony entries to lists of lists of floats
    best_harmony_list = [eval(harmony) for harmony in best_harmony_matches]
    type_matches = [eval(t) for t in type_matches]
    # print(best_harmony_list) 
    return best_harmony_list, type_matches
