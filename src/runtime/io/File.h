//
// Created by saeed on 04.11.23.
//

#ifndef CATDB_FILE_H
#define CATDB_FILE_H

#include <stdio.h>
#include <stdlib.h>

struct File {
    FILE *identifier;
    unsigned long pos;
    unsigned long read;
};

inline struct File *openMemFile(FILE *ident){
    struct File *f = (struct File *)malloc(sizeof(struct File));

    f->identifier = ident;
    f->pos = 0;

    return f;
}

inline struct File *openFile(const char *filename) {
    struct File *f = (struct File *)malloc(sizeof(struct File));

    f->identifier = fopen(filename, "r");
    f->pos = 0;

    if (f->identifier == NULL)
        return NULL;
    return f;
}

inline struct File *openFileForWrite(const char *filename) {
    struct File *f = (struct File *)malloc(sizeof(struct File));

    f->identifier = fopen(filename, "w+");
    f->pos = 0;

    if (f->identifier == NULL)
        return NULL;
    return f;
}

inline void closeFile(File *f) { fclose(f->identifier); }

inline char *getLine(File *f) {
    char *line = NULL;
    size_t len = 0;

    f->read = getline(&line, &len, f->identifier);
    f->pos += f->read;

    return line;
}

#endif //CATDB_FILE_H
