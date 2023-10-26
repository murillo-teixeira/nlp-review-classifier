import openai

# Set your API key
openai.api_key = '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'

# Replace 'your-model-id' with your fine-tuned model ID
model_id = 'ft:gpt-3.5-turbo-0613:personal::8CB1fZAX'

questions = []

with open('test_just_reviews.txt', 'r') as txtfile:
    for line in txtfile:
        questions.append(line)

# Create an empty list to store the responses
responses = []

# Loop through the list of questions and generate responses
for question in questions:
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a system that distinguishes between truthful and deceptive hotel reviews, and determines their polarity. It classifies reviews according with labels TRUTHFULPOSITIVE, TRUTHFULNEGATIVE, DECEPTIVEPOSITIVE, and DECEPTIVENEGATIVE."},
            {"role": "user", "content": question}
        ]
    )
    
    # Extract and store the assistant's reply
    assistant_reply = response['choices'][0]['message']['content']
    responses.append(assistant_reply)

# Print or process the responses as needed
for i, response in enumerate(responses):
    print(f"Question {i + 1}: {questions[i]}")
    print(f"Answer {i + 1}: {response}")
    print()

# You can also save the responses to a file if needed
with open("responses.txt", "w") as file:
    for i, response in enumerate(responses):
        file.write(f"Question {i + 1}: {questions[i]}\n")
        file.write(f"Answer {i + 1}: {response}\n\n")
