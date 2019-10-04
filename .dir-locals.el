;; First (minimal) attempt at configuring Emacs CC mode for the BLIS
;; layout requirements.
((c-mode . ((c-file-style . "stroustrup")
            (c-basic-offset . 4)
            (comment-start . "// ")
            (comment-end . "")
            (indent-tabs-mode . t)
            (tab-width . 4)
            (parens-require-spaces . nil))))
