CFLAGS = -Wall -Werror -g  -Wno-error -lm
CC = gcc $(CFLAGS)
SHELL = /bin/bash

all: main

clean:
	rm -f list_main *.o

# # 'make zip' to create zip file for submission
# zip: clean clean-tests
# 	rm -f $(AN)-code.zip
# 	cd .. && zip "$(CWD)/$(AN)-code.zip" -r "$(CWD)" -x "$(CWD)/test_cases/*" "$(CWD)/testius"
# 	@echo Zip created in $(AN)-code.zip
# 	@if (( $$(stat -c '%s' $(AN)-code.zip) > 10*(2**20) )); then echo "WARNING: $(AN)-code.zip seems REALLY big, check there are no abnormally large test files"; du -h $(AN)-code.zip; fi
# 	@if (( $$(unzip -t $(AN)-code.zip | wc -l) > 256 )); then echo "WARNING: $(AN)-code.zip has 256 or more files in it which may cause submission problems"; fi

filestats.o: filestats.c
	$(CC) -c filestats.c

main.o: main.c filestats.c
	$(CC) -c main.c

main: main.o filestats.o
	$(CC) -o main main.o filestats.o
