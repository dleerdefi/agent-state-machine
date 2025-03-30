import os
import re
from pathlib import Path

def find_files_with_utc_import(directory):
    files_to_fix = []
    
    # Regular expression pattern to find the import
    utc_import_pattern = re.compile(r'from\s+datetime\s+import\s+.*?\bUTC\b')
    
    # Walk through the directory tree
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if utc_import_pattern.search(content):
                            files_to_fix.append(filepath)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return files_to_fix

def fix_utc_imports(directory):
    files_to_fix = find_files_with_utc_import(directory)
    
    print(f"Found {len(files_to_fix)} files to fix:")
    for file in files_to_fix:
        print(f" - {file}")
    
    # Ask for confirmation
    if input("\nProceed with fixing these files? (y/n): ").lower() != 'y':
        print("Operation cancelled.")
        return
    
    for filepath in files_to_fix:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace import statement
            content = re.sub(
                r'from\s+datetime\s+import\s+(.*?)\bUTC\b(.*)',
                r'from datetime import \1timezone\2',
                content
            )
            
            # Replace UTC usage with timezone.utc
            content = re.sub(r'\bUTC\b(?!\s*=)', r'timezone.utc', content)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Fixed {filepath}")
        
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")

if __name__ == "__main__":
    src_directory = Path(__file__).parent.parent  # Assuming script is in src/scripts
    fix_utc_imports(src_directory)
    
    print("\nDone. Please check the files manually to ensure all replacements are correct.")
    print("Remember to test the code after making these changes.") 