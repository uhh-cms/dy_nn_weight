#!/usr/bin/env bash

setup_dyw() {
    # Runs the entire project setup, leading to a collection of environment variables starting with
    # "DYW_" (for Drell-Yan Weight project :) ), the installation of the software stack via virtual environments.

    #
    # prepare local variables
    #

    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local orig="${PWD}"
    local micromamba_url="https://micro.mamba.pm/api/micromamba/linux-64/latest"
    local pyv="3.11"
    local on_maxwell="$( [[ "$( hostname )" = max-*.desy.de ]] && echo "true" || echo "false" )"
    local env_suffix="$( ${on_maxwell} && echo "_maxwell" || echo "" )"
    ${on_maxwell} && echo "detected maxwell"


    #
    # global variables
    #

    # start exporting variables
    export DYW_BASE="${this_dir}"
    export DYW_USER="$( whoami )"
    export DYW_DATA_BASE="${DYW_DATA_BASE:-/data/dust/user/${DYW_USER}/dyw_data}"
    export DYW_SOFTWARE_BASE="${DYW_SOFTWARE_BASE:-${DYW_DATA_BASE}/software${env_suffix}}"
    export DYW_CONDA_BASE="${DYW_CONDA_BASE:-${DYW_SOFTWARE_BASE}/conda}"
    export DYW_VENV_BASE="${DYW_VENV_BASE:-${DYW_SOFTWARE_BASE}/venvs}"

    # external variables
    export LANGUAGE="${LANGUAGE:-en_US.UTF-8}"
    export LANG="${LANG:-en_US.UTF-8}"
    export LC_ALL="${LC_ALL:-en_US.UTF-8}"
    export PYTHONWARNINGS="ignore"
    export VIRTUAL_ENV_DISABLE_PROMPT="${VIRTUAL_ENV_DISABLE_PROMPT:-1}"
    export GLOBUS_THREAD_MODEL="none"
    export X509_USER_PROXY="/tmp/x509up_u$( id -u )"
    export X509_CERT_DIR="/cvmfs/grid.cern.ch/etc/grid-security/certificates"
    export X509_VOMS_DIR="/cvmfs/grid.cern.ch/etc/grid-security/vomsdir"
    export X509_VOMSES="/cvmfs/grid.cern.ch/etc/grid-security/vomses"
    export VOMS_USERCONF="${X509_VOMSES}"
    export CAPATH="${X509_CERT_DIR}"
    export MAMBA_ROOT_PREFIX="${DYW_CONDA_BASE}"
    export MAMBA_EXE="${MAMBA_ROOT_PREFIX}/bin/micromamba"


    #
    # minimal local software setup
    #

    ulimit -s unlimited

    # empty the PYTHONPATH and LD_LIBRARY_PATH
    export PYTHONPATH=""
    export LD_LIBRARY_PATH=""

    # update paths
    export PATH="${DYW_BASE/bin}:${PATH}"
    export PYTHONPATH="${DYW_BASE}:${DYW_CONDA_BASE}/lib/python${pyv}/site-packages"

    # conda base environment
    local conda_missing="$( [ -d "${DYW_CONDA_BASE}" ] && echo "false" || echo "true" )"
    if ${conda_missing}; then
        echo "installing conda/micromamba at ${DYW_CONDA_BASE}"
        (
            mkdir -p "${DYW_CONDA_BASE}"
            cd "${DYW_CONDA_BASE}"
            curl -Ls "${micromamba_url}" | tar -xvj -C . "bin/micromamba"
            ./bin/micromamba shell hook -y --root-prefix="${DYW_CONDA_BASE}" &> "micromamba.sh"
            mkdir -p "etc/profile.d"
            mv "micromamba.sh" "etc/profile.d"
            cat << EOF > ".mambarc"
changeps1: false
always_yes: true
channels:
  - conda-forge
EOF
        )
    fi

    # initialize conda
    source "${DYW_CONDA_BASE}/etc/profile.d/micromamba.sh" "" || return "$?"
    micromamba activate || return "$?"
    echo "initialized conda/micromamba"

    # install packages initially
    if ${conda_missing}; then
        echo
        echo "setting up conda/micromamba environment"

        # conda packages (nothing so far)
        micromamba install \
            libgcc \
            bash \
            zsh \
            "python=${pyv}" \
            git \
            git-lfs \
            || return "$?"
        micromamba clean --yes --all

        # update python base packages
        pip install --no-cache-dir -U \
            pip \
            setuptools \
            wheel \
            || return "$?"

        # additional packages
        if [ -f "${DYW_BASE}/requirements.txt" ]; then
            pip install --no-cache-dir -U -r "${DYW_BASE}/requirements.txt" || return "$?"
        fi
    fi

    # success
    return "0"
}

setup_dyw "$@"
