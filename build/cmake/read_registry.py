##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

# Import modules
import os
import sys
import re

def canonicalize_ws(str):
    # Remove leading and trailing whitespace.
    str = str.strip()
    # Remove duplicate spaces between words.
    res = " ".join(str.split())
    # Update the input argument.
    return res


def is_singleton(str):
    rval = False
    count_str = " "
    for item in str.split():
        count_str = count_str + "x"
    if count_str == "x":
        rval = True
    return rval


def is_singleton_family(familyname, memberlist):
    rval = False
    if is_singleton(memberlist):
        if memberlist == familyname:
            rval = True
    return rval


def is_in_list(word, str):
    rval = False
    for item in str.split():
        if item == word:
            rval = True
            break
    return rval


def assign_key_value(array, key, value):
    array.update({key: value})


def query_array(array, key):
    value = array.get(key)
    return value


def remove_from_list(strike_words, list):
    flist = ""
    for item in list.split():
        # Filter out any list item that matches any of the strike words.
        if not is_in_list(item, strike_words):
            flist = " ".join([flist, item])
    flist = canonicalize_ws(flist)
    # Return the filtered list.
    return flist

def replace_curconfig_configset(klisttmp, curconfig, configset):
    tmplist = list(klisttmp.split(" "))
    ind = tmplist.index(curconfig)
    tmplist.remove(curconfig)
    tmplist.insert(ind, configset)
    newlist = " ".join(map(str, tmplist))
    return newlist

def rm_duplicate_words(str):
    res = " ".join(str.split()[::-1])
    res = " ".join(dict.fromkeys(res.split()))
    str = " ".join(res.split()[::-1])
    return str

def pass_config_kernel_registries(filename, passnum):
    global config_blist
    global indirect_blist
    global config_registry
    global kernel_registry
    # first argument: the file containing the configuration registry.
    # second argument: the pass number: 0 or 1. Pass 0 builds the
    # indirect config blacklist (indirect_blist) ONLY. Pass 1 actually
    # begins populating the config and kernel registries, and assumes
    # the indirect_blist has already been created.
    # Initialize a list of indirect blacklisted configurations for the
    # current iteration. These are configurations that are invalidated by
    # the removal of blacklisted configurations. For example, if haswell
    # is registered as needing the 'haswell' and 'zen' kernel sets:
    #    haswell: haswell/haswell/zen
    # and 'zen' was blacklisted because of the compiler version, then the
    # 'haswell' configuration must be omitted from the registry, as it no
    # longer has all of the kernel sets it was expecting.
    if passnum == 0:
        indirect_blist = ""
    # For convenience, merge the original and indirect blacklists.
    # NOTE: During pass 0, all_blist is equal to config_blist, since
    # indirect_blist is still empty.
    all_blist = config_blist + indirect_blist
    # Disable support for indirect blacklisting by returning early during
    # pass 0. See issue #214 for details [1]. Basically, I realized that
    # indirect blacklisting is not needed in the use case that I envisioned
    # in the real-life example above. If a subconfiguration such as haswell
    # is defined to require the zen kernel set, it implies that the zen
    # kernels can be compiled with haswell compiler flags. That is, just
    # because the zen subconfig (and its compiler flags) is blacklisted
    # does not mean that the haswell subconfig cannot compile the zen
    # kernels with haswell-specific flags.
    # [1] https://github.com/flame/blis/issues/214
    if passnum == 0:
        return

    cfg = open(filename, "r+")
    while True:
        line = cfg.readline()
        if not line:
            break

        # We've stripped out leading whitespace and trailing comments. If
        # the line is now empty, then we can skip it altogether.
        if re.match(r'\n', line) or re.match(r'#', line):
            continue

        # Read the config name and config list for the current line.
        cname, list = line.split(':')
        cname = cname.strip()
        list = list.strip()
        # If we encounter a slash, it means the name of the configuration
        # and the kernel set needed by that configuration are different.
        if list.find("/") != -1:
            clist = ""
            klist = ""
            # The sub-configuration name is always the first sub-word in
            # the slash-separated compound word.
            # Delete the sub-configuration name from the front of the
            # string, leaving the slash-separated kernel names (or just
            # the kernel name, if there is only one).
            # Replace the slashes with spaces to transform the string
            # into a space-separated list of kernel names.
            list = list.replace("/", " ")
            config, kernels = list.split(" ", 1)

            clist = clist + config
            klist = klist + kernels
        else:
            clist = list
            klist = list

        # Strip out whitespace from the config name and config/kernel list
        # on each line.
        cname = canonicalize_ws(cname)
        clist = canonicalize_ws(clist)
        klist = canonicalize_ws(klist)
        # Next, we prepare to:
        # - pass 0: inspect klist for blacklisted configurations, which may
        #   reveal configurations as needing to be indirectly blacklisted.
        # - pass 1: compare cname to the blacklists and commit clist/klist
        #   to their respective registries, as appropriate.
        # Handle singleton and umbrella configuration entries separately.
        if is_singleton_family(cname, clist):
            # Singleton configurations/families.
            # Note: for singleton families, clist contains one item, which
            # always equals cname, but klist could contain more than one
            # item.
            # Only consider updating the indirect blacklist (pass 0) or
            # committing clist and klist to the registries (pass 1) if the
            # configuration name (cname) is not blacklisted.
            if not is_in_list(cname, all_blist):
                if passnum == 0:
                    # Even if the cname isn't blacklisted, one of the requisite
                    # kernels might be, so we need to check klist for blacklisted
                    # items. If we find one, we must assume that the entire entry
                    # must be thrown out. (Ideally, we would simply fall back to
                    # reference code for the blacklisted kernels, but that is not
                    # at all straightforward under the current configuration
                    # system architecture.) Thus, we add cname to the indirect
                    # blacklist.
                    for item in klist.split():
                        if is_in_list(item, config_blist):
                            indirect_blist = indirect_blist + cname
                            break
                if passnum == 1:
                    # Store the clist to the cname key of the config registry.
                    # config_registry[${cname}]=${clist}
                    assign_key_value(config_registry, cname, clist)
            if passnum == 1:
                # Store the klist to the cname key of the kernel registry.
                # kernel_registry[${cname}]=${klist}
                assign_key_value(kernel_registry, cname, klist)
        else:
            # Umbrella configurations/families.
            # First we check cname, which should generally not be blacklisted
            # for umbrella families, but we check anyway just to be safe.
            if not is_in_list(cname, all_blist):
                if passnum == 1:
                    # Check each item in the clist and klist. (At this point,
                    # clist == klist.) If any sub-config is blacklisted, we
                    # omit it from clist and klist.
                    for item in clist.split():
                        if is_in_list(item, all_blist):
                            clist = remove_from_list(item, clist)
                            klist = remove_from_list(item, klist)
                    # Store the config and kernel lists to entries that
                    # corresponds to the config name.
                    assign_key_value(config_registry, cname, clist)
                    assign_key_value(kernel_registry, cname, klist)
    cfg.close()
    if passnum == 0:
        # Assign the final indirect blacklist (with whitespace removed).
        indirect_blist = canonicalize_ws(indirect_blist)


def read_registry_file(filename):
    global config_registry
    global kernel_registry
    # Execute an initial pass through the config_registry file so that
    # we can accumulate a list of indirectly blacklisted configurations,
    # if any.
    pass_config_kernel_registries(filename, 0)
    # Now that the indirect_blist has been created, make a second pass
    # through the 'config_registry' file, this time creating the actual
    # config and kernel registry data structures.
    pass_config_kernel_registries(filename, 1)
    # Now we must go back through the config_registry and subsitute any
    # configuration families with their constituents' members. Each time
    # one of these substitutions occurs, we set a flag that causes us to
    # make one more pass. (Subsituting a singleton definition does not
    # prompt additional iterations.) This process stops when a full pass
    # does not result in any subsitution.

    iterate_again = 1
    while iterate_again == 1:
        iterate_again = 0
        for cr_var in config_registry:
            config = cr_var
            clist = query_array(config_registry, config)
            # The entries that define singleton families should never need any substitution.
            if is_singleton_family(config, clist):
                continue
            for mem in clist.split():
                mems_mem = query_array(config_registry, mem)
                # If mems_mem is empty string, then mem was not found as a key
                # in the config list associative array. In that case, we continue
                # and will echo an error later in the script.
                if not (mems_mem and mems_mem.strip()):
                    continue
                if mem != mems_mem:
                    clist = query_array(config_registry, config)
                    # Replace the current config with its constituent config set,
                    # canonicalize whitespace, and then remove duplicate config
                    # set names, if they exist. Finally, update the config registry
                    # with the new config list.
                    #newclist = replace_curconfig_configset(clist, mem, mems_mem)
                    newclist = re.sub(r"\b{}\b".format(mem), mems_mem, clist)
                    newclist = canonicalize_ws(newclist)
                    newclist = rm_duplicate_words(newclist)
                    assign_key_value(config_registry, config, newclist)
                    # Since we performed a substitution and changed the config
                    # list, mark the iteration flag to continue another round,
                    # but only if the config (mem) value is NOT present
                    # in the list of sub-configs. If it is present, then further
                    # substitution may not necessarily be needed this round.
                    if not is_in_list(mem, mems_mem):
                        iterate_again = 1
    # Similar to what we just did for the config_registry, we now iterate
    # through the kernel_registry and substitute any configuration families
    # in the kernel list (right side of ':') with the members of that
    # family's kernel set. This process continues iteratively, as before,
    # until all families have been replaced with singleton configurations'
    # kernel sets.
    iterate_again = 1
    while iterate_again == 1:
        iterate_again = 0
        for kr_var in kernel_registry:
            config = kr_var
            klist = query_array(kernel_registry, config)
            # The entries that define singleton families should never need
            # any substitution. In the kernel registry, we know it's a
            # singleton entry when the cname occurs somewhere in the klist.
            # (This is slightly different than the same test in the config
            # registry, where we test that clist is one word and that
            # clist == cname.)
            if is_in_list(config, klist):
                # echo "debug: '${config}' not found in '${klist}'; skipping."
                continue
            for ker in klist.split():
                kers_ker = query_array(kernel_registry, ker)
                # If kers_ker is empty string, then ker was not found as a key
                # in the kernel registry. While not common, this can happen
                # when ker identifies a kernel set that does not correspond to
                # any configuration. (Example: armv7a and armv8a kernel sets are
                # used by cortexa* configurations, but do not correspond to their
                # own configurations.)
                if not (kers_ker and kers_ker.strip()):
                    continue
                # If the current config/kernel (ker) differs from its singleton kernel
                # entry (kers_ker), then that singleton entry was specified to use
                # a different configuration's kernel set. Thus, we need to replace the
                # occurrence in the current config/kernel name with that of the kernel
                # set it needs.
                if ker != kers_ker:
                    klisttmp = query_array(kernel_registry, config)
                    # Replace the current config with its requisite kernels,
                    # canonicalize whitespace, and then remove duplicate kernel
                    # set names, if they exist. Finally, update the kernel registry
                    # with the new kernel list.
                    #newklist = replace_curconfig_configset(klisttmp, ker, kers_ker)
                    newklist = re.sub(r"\b{}\b".format(ker), kers_ker, klisttmp)
                    newklist = canonicalize_ws(newklist)
                    newklist = rm_duplicate_words(newklist)
                    assign_key_value(kernel_registry, config, newklist)
                    # Since we performed a substitution and changed the kernel
                    # list, mark the iteration flag to continue another round,
                    # unless we just substituted using a singleton family
                    # definition, in which case we don't necessarily need to
                    # iterate further this round.
                    if not is_in_list(ker, kers_ker):
                        iterate_again = 1


def build_kconfig_registry(familyname):
    global config_registry
    global kernel_registry
    global kconfig_registry
    clist = query_array(config_registry, familyname)
    for config in clist.split():
        # Look up the kernels for the current sub-configuration.
        kernels = query_array(kernel_registry, config)
        for kernel in kernels.split():
            # Add the sub-configuration to the list associated with the kernel.
            # Query the current sub-configs for the current ${kernel}.
            cur_configs = query_array(kconfig_registry, kernel)
            # Add the current sub-configuration to the list of sub-configs we just queried.
            if cur_configs and cur_configs.strip():
                cur_configs = " ".join([cur_configs, config])
                cur_configs = cur_configs.strip()
            else:
                cur_configs = config
            newvalue = canonicalize_ws(cur_configs)
            # Update the array.
            assign_key_value(kconfig_registry, kernel, newvalue)


def lastWord(string):
    # finding the index of last space
    index = string.rfind(" ")
    # last word
    return string[index + 1:]



config_blist = ""
indirect_blist = ""
config_registry = {}
kernel_registry = {}
kconfig_registry = {}

def process_config():
    # Obtain the script name.
    cwd = os.getcwd()
    path, arch = os.path.split(sys.argv[1])
    target_file = os.path.join(sys.argv[2], 'config_registry')

    read_registry_file(target_file)

    config_list = query_array(config_registry, arch)
    kernel_list = query_array(kernel_registry, arch)

    build_kconfig_registry(arch)

    config_list = " ".join(config_list.split())
    kernel_list = " ".join(kernel_list.split())

    # We use a sorted version of kernel_list so that it ends up matching the
    # display order of the kconfig_registry above.
    kernel_list_sort = kernel_list

    kconfig_map = ""
    for kernel in kernel_list_sort.split():
        configs = query_array(kconfig_registry, kernel)

        has_one_kernel = is_singleton(configs)
        contains_kernel = is_in_list(kernel, configs)

        # Check if the list is a singleton.
        if has_one_kernel:
            reducedclist = configs
        # Check if the list contains a sub-config name that matches the kernel.
        elif contains_kernel:
            reducedclist = kernel
        # Otherwise, use the last name.
        else:
            last_config = lastWord(configs)
            reducedclist = last_config

        # Create a new "kernel:subconfig" pair and add it to the kconfig_map
        # list, removing whitespace.
        new_pair = kernel+':'+reducedclist
        kconfig_map = " ".join([kconfig_map, new_pair])
        kconfig_map = canonicalize_ws(kconfig_map)

    config = " ; ".join([config_list, kernel_list, kconfig_map])
    return config


# Function call for config family names
CONFIG = process_config()
print(CONFIG)
