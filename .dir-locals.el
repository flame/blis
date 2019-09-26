;; First (minimal) attempt at configuring Emacs CC mode for the BLIS
;; layout requirements.
((nil . ((indent-tabs-mode . t)
	 (tab-width . 4)
	 (parens-require-spaces . nil)))
 (c-mode . ((c-file-style . "stroustrup")
	    (c-basic-offset . 4)
            (subdirs . nil))))
