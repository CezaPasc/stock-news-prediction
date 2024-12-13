#!/bin/bash

declare -A secret_descriptions
declare -A path_descriptions
declare -A is_file

secret_descriptions=(
    ["SERP_API_KEY"]="API key for the serp search engine service"
    ["GOOGLE_API_KEY"]="API key for the goggle search engine service"
    ["GOOGLE_ENGINE_ID"]="Identifier of a custom goggle search engine"
    ["REDDIT_CLIENT_ID"]="Client ID for the Reddit API"
    ["REDDIT_CLIENT_SECRET"]="Secret for the Reddit API"
    ["HUGGINGFACE_TOKEN"]="API Token for HuggingFace (used for accessing modes like Llama for augmenting data)"
    ["PRICE_ONLY_CACHE"]="Use only already cached prices. No requests will be made to AlphaVantage"
)

path_descriptions=(
    ["DATA"]="The directory where the datasets will be saved"
    ["PRICE_CACHE"]="The directory for storing the cached intraday data"
    ["ASSETS"]="A JSON file containing a selection of assets mapping their names to their ticker names"
    ["ASSETS_NAMING"]="A dictonary renaming ticker names, used for avoiding to have two different ticker names for the same company (GOOG, GOOGL)"
    ["TWITTER_USERS"]="path for alist of URLs to profiles of X"
    ["RISK_FREE_RATES"]="A CSV file containing risk-free rates"
)

is_file=(
    ["DATA"]=false
    ["PRICE_CACHE"]=false
    ["ASSETS"]=true
    ["ASSETS_NAMING"]=true
)


# Function to get the value for a specific key from the .env file
get_env_value() {
    local key="$1"
    local value=$(grep -E "^$key=" .env.bu | cut -d '=' -f 2)

    # If the value is empty (key doesn't exist), return an empty string
    echo "${value:-}"
}


# Function to prompt for value and validate path
function prompt_for_path() {
    local var_name="$1"
    local description="${path_descriptions[$var_name]}"
    local is_file_flag="${is_file[$var_name]}"

    echo "Enter value for $var_name ($description)"

    result=$(get_env_value "$var_name")
    if [[ -n "$result" ]]; then
        echo "Old value: $result"
        read -p "Do you want to redefine it? (yes/no): " redefine
        if [[ "$redefine" != "yes" ]]; then
            echo "$var_name=$result" >> .env
            return
        fi
    fi

    read -p "$var_name=" value

    # Validate path existence
    if [[ ! -e "$value" ]]; then
        if [[ "$is_file_flag" == true ]]; then
            echo "File does not exist. Please provide a valid file path."
            prompt_for_path "$var_name"  # Call function recursively
            return
        else
            read -p "Path does not exist. Create it? (yes/no): " create_path
            if [[ "$create_path" == "yes" ]]; then
                mkdir -p "$value"
            else
                echo "Please provide a valid path."
                prompt_for_path "$var_name"  # Call function recursively
                return
            fi
        fi
    fi

    # Append to .env file
    echo "$var_name=$value" >> .env
}

# Function to prompt for secrets
function prompt_for_secret() {
    local var_name="$1"
    local description="${secret_descriptions[$var_name]}"

    echo "Enter value for $var_name ($description)"

    result=$(get_env_value "$var_name")
    if [[ -n "$result" ]]; then
        echo "Old value: $result"
        read -p "Do you want to redefine it? (yes/no): " redefine
        if [[ "$redefine" != "yes" ]]; then
            echo "$var_name=$result" >> .env
            return
        fi
    fi

    read -p "$var_name=" value


    # Append to .env file
    echo "$var_name=$value" >> .env
}


# Create or update .env file
if [[ -e .env ]]; then
    mv .env .env.bu
else
    touch .env.bu
fi

> .env


# Iterate through environment variables
for var_name in "${!path_descriptions[@]}"; do
    prompt_for_path "$var_name"
    echo
done

for var_name in "${!secret_descriptions[@]}"; do
    prompt_for_secret "$var_name"
    echo
done
