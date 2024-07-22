import openai
import pandas as pd

# Load your CSV file
df = pd.read_csv('locations.csv')

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Function to generate a prompt and get facts from OpenAI API
def get_facts(row):
    # Base prompt
    prompt = f"Provide detailed facts about {row['Location']}."

    # Add additional information based on available columns
    if 'Country' in row:
        prompt += f" It is located in {row['Country']}."
    if 'Population' in row:
        prompt += f" It has a population of {row['Population']}."
    if 'Landmarks' in row:
        landmarks = row['Landmarks']
        prompt += f" Notable landmarks include {landmarks}."

    # Fetch facts from OpenAI
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Iterate through the DataFrame and get facts for each location
facts = []
for _, row in df.iterrows():
    fact = get_facts(row)
    facts.append(fact)

# Add the facts to the DataFrame
df['Facts'] = facts

# Save the new DataFrame to a CSV file
df.to_csv('locations_with_facts.csv', index=False)

# Display the DataFrame to the user
import ace_tools as tools; tools.display_dataframe_to_user(name="Location Facts", dataframe=df)
