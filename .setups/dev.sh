action() {
    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local hbt_dir="$( cd "$( dirname "${this_dir}" )" && pwd )"

    local on_maxwell="$( [[ "$( hostname )" = max-*.desy.de ]] && echo "true" || echo "false" )"
    local env_suffix="$( ${on_maxwell} && echo "_maxwell" || echo "_naf" )"
    ${on_maxwell} && echo "detected maxwell"

    # cf variables
    export CF_NAF_USER="${CF_NAF_USER:-$( whoami )}"
    export CF_CERN_USER="${CF_CERN_USER:-$( whoami )}"
    export CF_CERN_USER_FIRSTCHAR="${CF_CERN_USER:0:1}"
    export CF_DATA="/data/dust/user/${CF_NAF_USER}/dyw_data"
    export CF_SOFTWARE_BASE="${CF_DATA}/software${env_suffix}"
    export CF_JOB_BASE="${CF_DATA}/jobs"
    export CF_STORE_NAME="dyw_store"
    export CF_STORE_LOCAL="${CF_DATA}/dyw_store"
    export CF_WLCG_CACHE_ROOT="${CF_DATA}/dyw_cache"
    export CF_WLCG_USE_CACHE="true"
    export CF_WLCG_CACHE_CLEANUP="false"
    export CF_VENV_SETUP_MODE_UPDATE="false"
    export CF_VENV_SETUP_MODE="update"
    export CF_LOCAL_SCHEDULER="true"
    export CF_SCHEDULER_HOST=""
    export CF_SCHEDULER_PORT=""
    export CF_FLAVOR="cms"
    export HBT_ON_MAXWELL="${on_maxwell}"

    ${on_maxwell} && export LAW_TARGET_TMP_DIR="${CF_DATA}/dyw_tmp"

    # akways skip tmp dir check for now
    export CF_SKIP_CHECK_TMP_DIR="true"

    # skip initial law indexing
    export CF_SKIP_LAW_INDEX="true"

    return "0"
}
action "$@"
