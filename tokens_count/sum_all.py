import pandas as pd

path = "tokens_count/total_tokens.csv"

df = pd.read_csv(path)

# cols: date_time,agent_name,prompt,generated_answer,input tokens,output tokens
# seperate sum by 'Embedded' and not 'Embedded'
# seperate sum by input and output

print("\n ----- Tokens -------- \n")

embedded_df = df[df['agent_name'] == 'Embedded']
not_embedded_df = df[df['agent_name'] != 'Embedded']

# print total input and total ouput for each df
total_input_embedded = embedded_df["input tokens"].sum()
total_output_embedded = embedded_df["output tokens"].sum()
print(f"Embedded \n Total input tokens: {total_input_embedded} \n Total output tokens: {total_output_embedded}")

total_input_not_embedded = not_embedded_df["input tokens"].sum()
total_output_not_embedded = not_embedded_df["output tokens"].sum()
print(f"Not Embedded \n Total input tokens: {total_input_not_embedded} \n Total output tokens: {total_output_not_embedded}")

print("\n ----- Prices -------- \n")

# Prices for not embedded
PRICE_4o_1M_INPUT_TOKENS = 2.5
PRICE_4o_1M_OUTPUT_TOKENS = 10

# prices for embedded
PRICE_EMBED_SMALL_1M_INPUT_TOKENS = 0.02

# Print how much cost
not_embedded_cost = (total_input_not_embedded / 1_000_000) * PRICE_4o_1M_INPUT_TOKENS + (total_output_not_embedded / 1_000_000) * PRICE_4o_1M_OUTPUT_TOKENS
embedded_cost = (total_input_embedded / 1_000_000) * PRICE_EMBED_SMALL_1M_INPUT_TOKENS

print(f"Not Embedded - Total cost: {not_embedded_cost}")
print(f"Embedded - Total cost: {embedded_cost}")

print("\n ----- Total $$$ -------- \n")
# print rounded 2, sum both
both = round(not_embedded_cost + embedded_cost, 2)
print(f"Total cost (both): {both}")

# save all these print to txt file
with open("tokens_count/total_cost.txt", "w") as f:
    f.write(f"Not Embedded - Total cost: {not_embedded_cost}\n")
    f.write(f"Embedded - Total cost: {embedded_cost}\n")
    f.write(f"Total cost (both): {both}\n")