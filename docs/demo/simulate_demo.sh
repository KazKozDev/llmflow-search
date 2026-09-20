#!/bin/zsh
set -euo pipefail

blue=$'\033[38;2;103;148;254m'
flow_blue=$'\033[38;2;81;115;205m'
search_gray=$'\033[38;2;174;181;193m'
reset=$'\033[0m'
bold=$'\033[1m'
blue_head=$'\033[38;2;112;143;181m'

print -r -- "
${blue}
██╗     ██╗     ███╗   ███╗ ${flow_blue}███████╗██╗      ██████╗ ██╗    ██╗
██║     ██║     ████╗ ████║ ${flow_blue}██╔════╝██║     ██╔═══██╗██║    ██║
██║     ██║     ██╔████╔██║ ${flow_blue}█████╗  ██║     ██║   ██║██║ █╗ ██║
██║     ██║     ██║╚██╔╝██║ ${flow_blue}██╔══╝  ██║     ██║   ██║██║███╗██║
███████╗███████╗██║ ╚═╝ ██║ ${flow_blue}██║     ███████╗╚██████╔╝╚███╔███╔╝
╚══════╝╚══════╝╚═╝     ╚═╝ ${flow_blue}╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝${reset}

${search_gray}███████╗ ███████╗  █████╗  ██████╗   ██████╗ ██╗  ██╗
██╔════╝ ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔════╝ ██║  ██║
███████╗ █████╗   ███████║ ██████╔╝ ██║      ███████║
╚════██║ ██╔══╝   ██╔══██║ ██╔══██╗ ██║      ██╔══██║
███████║ ███████╗ ██║  ██║ ██║  ██║ ╚██████╗ ██║  ██║
╚══════╝ ╚══════╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝  ╚═════╝ ╚═╝  ╚═╝${reset}

"

sleep 0.8

print -r -- "${blue_head}${bold}Ollama models (4 total):${reset}"
print -r -- ""
print -r -- "  ${blue_head}${bold} 1.${reset} gemma3:12b                           7.5 GB"
print -r -- "  ${blue_head}${bold} 2.${reset} gemma4:26b-mlx                       16.4 GB"
print -r -- "  ${blue_head}${bold} 3.${reset} qwen2.5:7b                            4.7 GB"
print -r -- "  ${blue_head}${bold} 4.${reset} qwen3:8b                              4.9 GB"
print -r -- ""
printf "Pick model number [Enter = gemma4:26b-mlx] > "
