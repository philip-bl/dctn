;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")
((python-mode
  (python-shell-interpreter . "ipython")
  (python-shell-interpreter-args . "-i --simple-prompt -c \"%autoreload 2\"")
  (eval add-hook 'before-save-hook 'elpy-format-code nil t)))
