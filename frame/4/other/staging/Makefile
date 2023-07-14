

.PHONY: clean

all: f2c

f2c:
	@- ./preprocess.sh
	@- ./run-f2c.sh
	@- ./postprocess.sh
	@- ./canonicalize.sh
	@- ./postprocess2.sh
	@- ./postprocess3.sh
	@- ./postprocess4.sh
	@- ./create-prototypes.sh
	@- ./apply-namespace.sh

#	@- cat *.c > all.c

clean:
	@- rm -f *.c f2c_lapack.h


# Consider sed'ing bla_ in front of all lapack symbol names
# to avoid link-time conflict with a LAPACK library.





#recu-sed.sh -c "s/, (ftnlen)1, (ftnlen)1//g"
#recu-sed.sh -c "s/, (ftnlen)1, (/)/g"
#recu-sed.sh -c "s/ftnlen)1)//g"
#recu-sed.sh -c "s/, ($/)/g"
#recu-sed.sh -c "s/ftnlen)1, (//g"
#recu-sed.sh -c "s/, (ftnlen)6//g"

#recu-sed.sh -c "s/, ftnlen, ftnlen)/)/g"
#recu-sed.sh -c "s/, ftnlen)/)/g"

