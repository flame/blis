set(BLIS_FRAGMENT_IGNORE_DIRS
    attic
    broken
    old
    other
    temp
    tmp
    test
    p10_testsuite
)

function(blis_collect out root)
    set(all "")
    foreach(suf ${ARGN})
        file(GLOB_RECURSE found CONFIGURE_DEPENDS "${root}/*.${suf}")
        list(APPEND all ${found})
    endforeach()
    set(keep "")
    foreach(f ${all})
        file(RELATIVE_PATH rel "${root}" "${f}")
        set(skip FALSE)
        foreach(ig ${BLIS_FRAGMENT_IGNORE_DIRS})
            if(rel MATCHES "(^|/)${ig}(/|$)")
                set(skip TRUE)
                break()
            endif()
        endforeach()
        if(NOT skip)
            list(APPEND keep "${f}")
        endif()
    endforeach()
    set(${out} ${keep} PARENT_SCOPE)
endfunction()

# Directories containing header files under <root> (deduped) — used both for
# flatten-headers.py's search list and for -I flags.
function(blis_header_dirs out)
    set(dirs "")
    foreach(root ${ARGN})
        foreach(suf h hh hpp hxx)
            file(GLOB_RECURSE hdrs CONFIGURE_DEPENDS "${root}/*.${suf}")
            foreach(h ${hdrs})
                file(RELATIVE_PATH rel "${root}" "${h}")
                set(skip FALSE)
                foreach(ig ${BLIS_FRAGMENT_IGNORE_DIRS})
                    if(rel MATCHES "(^|/)${ig}(/|$)")
                        set(skip TRUE)
                        break()
                    endif()
                endforeach()
                if(skip)
                    continue()
                endif()
                get_filename_component(d "${h}" DIRECTORY)
                list(APPEND dirs "${d}")
            endforeach()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES dirs)
    set(${out} ${dirs} PARENT_SCOPE)
endfunction()
