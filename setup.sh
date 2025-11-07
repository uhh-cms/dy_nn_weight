#!/usr/bin/env bash

setup_dyw() {
    # Initializes the hbt analysis and sets up some environment variables on top.
    # Extra variables start with "DYW_" (for Drell-Yan Weight project :) ).

    #
    # prepare local variables
    #

    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local orig="${PWD}"
    local on_maxwell="$( [[ "$( hostname )" = max-*.desy.de ]] && echo "true" || echo "false" )"
    local env_suffix="$( ${on_maxwell} && echo "_maxwell" || echo "" )"
    ${on_maxwell} && echo "detected maxwell"


    #
    # check if the analysis is cloned, symlink setups
    #

    local HBT_DIR="${this_dir}/modules/hbt"
    if [ ! -d "${HBT_DIR}/hbt" ]; then
        >&2 echo "the hbt submodule in modules/hbt is not set up yet, run 'git submodule update --init --recursive modules/hbt' first"
        return "1"
    fi
    if [ ! -f "${HBT_DIR}/.setups/dev.sh" ]; then
        mkdir -p "${HBT_DIR}/.setups"
        ln -s "${this_dir}/.setups/dev.sh" "${HBT_DIR}/.setups/dev.sh"
    fi


    #
    # global variables
    #

    # start exporting variables
    export DYW_BASE="${this_dir}"
    export DYW_USER="$( whoami )"
    export DYW_DATA_BASE="${DYW_DATA_BASE:-/data/dust/user/${DYW_USER}/dyw_data}"
    export DYW_SOFTWARE_BASE="${DYW_SOFTWARE_BASE:-${DYW_DATA_BASE}/software${env_suffix}}"

    # update paths
    export PATH="${DYW_BASE/bin}:${PATH}"
    export PYTHONPATH="${DYW_BASE}:${PYTHONPATH}"


    #
    # trigger the analysis setup
    #

    source "${HBT_DIR}/setup.sh" "dev" || return "$?"


    # success
    return "0"
}

setup_dyw "$@"
