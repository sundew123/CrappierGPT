#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
struct codepoint {
	size_t numTokens;
	double sampleLog;
	double logTokenizations;
	double entropy;
	char valid;
	char content[5];
};
struct node {
	char* vocab;
	struct node* next;
};
long int generateSamplesFromString(FILE *dump, FILE *vocabFile, size_t maxLen, struct codepoint *wrappedString, size_t numCodepoints, struct node **vocab, size_t context, double scaling, long int *startPoint) {
	double *buffer = malloc(maxLen * (context - 1) * sizeof(double));
	if (buffer == NULL) {
		return -1;
	}
	char *checksum = malloc(maxLen * (context - 1) * sizeof(char));
	if (checksum == NULL) {
		free(buffer);
		return -1;
	}
	char *scratch = malloc(maxLen * sizeof(char));
	if (scratch == NULL) {
		free(buffer);
		free(checksum);
		return -1;
	}
	for (size_t i = 0; i < maxLen * (context - 1); i++) {
		buffer[i] = log2(0);
	}
	for (size_t i = 0; i < maxLen * (context - 1); i++) {
		checksum[i] = 0;
	}
	for (size_t i = 0; i < maxLen; i++) {
		scratch[i] = 0;
	}
	size_t queueStart = 0;
	wrappedString[0].sampleLog = log2(1);
	wrappedString[0].valid = 1;
	for (size_t i = 1; i < numCodepoints; i++) {
		queueStart = (i <= maxLen) ? 0 : ((queueStart + 1) % maxLen);
		char atomicExists = 0;
		struct node *lastVocab = NULL;
		for (struct node *j = *vocab; j != NULL; j = j->next) {
			lastVocab = j;
			size_t backDist = strlen(j->vocab);
			size_t step = 1;
			char match = 1;
			while (match && i >= step && backDist >= strlen(wrappedString[i - step].content)) {
				char temp = j->vocab[backDist];
				j->vocab[backDist] = '\0';
				if (strcmp(j->vocab + backDist - strlen(wrappedString[i - step].content), wrappedString[i - step].content)) {
					match = 0;
				}
				j->vocab[backDist] = temp;
				backDist -= strlen(wrappedString[i - step].content);
				step++;
			}
			if (backDist) {
				match = 0;
			}
			if (match) {
				scratch[maxLen + 1 - step] = 1;
				if (step == 2) {
					atomicExists = 1;
				}
			}
		}
		if (!atomicExists) {
			struct node *temp = malloc(sizeof(struct node));
			if (temp == NULL) {
				free(buffer);
				free(checksum);
				free(scratch);
				return -1;
			}
			temp->vocab = malloc((strlen(wrappedString[i - 1].content) + 1) * sizeof(char));
			if (temp->vocab == NULL) {
				free(buffer);
				free(checksum);
				free(scratch);
				free(temp);
				return -1;
			}
			strcpy(temp->vocab, wrappedString[i - 1].content);
			temp->next = NULL;
			if (lastVocab == NULL) {
				*vocab = temp;
			} else {
				lastVocab->next = temp;
				long int newVocPos = ftell(vocabFile);
				if (newVocPos < 0) {
					free(buffer);
					free(scratch);
					free(checksum);
					return -1;
				}
				if (newVocPos != 0) {
					if (fwrite(",", sizeof(char), 1, vocabFile) != 1) {
						free(buffer);
						free(scratch);
						free(checksum);
						return -1;
					}
				}
			}
			if (!strcmp(wrappedString[i - 1].content, "\"")) {
				if (fwrite("\"\"\"\"", sizeof(char), 4, vocabFile) != 4) {
					free(buffer);
					free(scratch);
					free(checksum);
					return -1;
				}
			} else if (!strcmp(wrappedString[i - 1].content, ",")) {
				if (fwrite("\",\"", sizeof(char), 3, vocabFile) != 3) {
					free(buffer);
					free(scratch);
					free(checksum);
					return -1;
				}
			} else if (!strcmp(wrappedString[i - 1].content, "\n")) {
				if (fwrite("\"\n\"", sizeof(char), 3, vocabFile) != 3) {
					free(buffer);
					free(scratch);
					free(checksum);
					return -1;
				}
			} else {
				if (fwrite(wrappedString[i - 1].content, sizeof(char), strlen(wrappedString[i - 1].content), vocabFile) != strlen(wrappedString[i - 1].content)) {
					free(buffer);
					free(scratch);
					free(checksum);
					return -1;
				}
			}
		}
		scratch[maxLen - 1] = 1;
		double maxLog = log2(0);
		for (size_t j = 0; j < maxLen; j++) {
			if (maxLen <= i + j && scratch[j]) {
				if (checksum[maxLen * (context - 2) + (queueStart + ((i > maxLen) ? j : (i + j - maxLen))) % maxLen]) {
					wrappedString[i].valid = 1;
				}
				if (buffer[maxLen * (context - 2) + (queueStart + ((i > maxLen) ? j : (i + j - maxLen))) % maxLen] > maxLog) {
					maxLog = buffer[maxLen * (context - 2) + (queueStart + ((i > maxLen) ? j : (i + j - maxLen))) % maxLen];
				}
			}
		}
		double sumLog = 0;
		for (size_t j = 0; j < maxLen; j++) {
			if (maxLen <= i + j && scratch[j]) {
				if (maxLog != log2(0)) {
					sumLog += pow(2, buffer[maxLen * (context - 2) + (queueStart + ((i > maxLen) ? j : (i + j - maxLen))) % maxLen] - maxLog);
				}
			}
		}
		if (maxLog != log2(0)) {
			wrappedString[i].sampleLog = maxLog + log2(sumLog);
		}
		for (size_t j = context - 2; j > 0; j--) {
			maxLog = log2(0);
			checksum[maxLen * j + (queueStart + ((i > maxLen) ? maxLen : i)) % maxLen] = 0;
			buffer [maxLen * j + (queueStart + ((i > maxLen) ? maxLen : i)) % maxLen] = log2(0);
			for (size_t k = 0; k < maxLen; k++) {
				if (maxLen <= i + k && scratch[k]) {
					if (checksum[maxLen * (j - 1) + (queueStart + ((i > maxLen) ? k : (i + k - maxLen))) % maxLen]) {
						checksum[maxLen * j + (queueStart + ((i > maxLen) ? maxLen : i)) % maxLen] = 1;
					}
					if (buffer[maxLen * (j - 1) + (queueStart + ((i > maxLen) ? k : (i + k - maxLen))) % maxLen] > maxLog) {
						maxLog = buffer[maxLen * (j - 1) + (queueStart + ((i > maxLen) ? k : (i + k - maxLen))) % maxLen];
					}
				}
			}
			sumLog = 0;
			for (size_t k = 0; k < maxLen; k++) {
				if (maxLen <= i + k && scratch[k]) {
					if (maxLog != log2(0)) {
						sumLog += pow(2, buffer[maxLen * (j - 1) + (queueStart + ((i > maxLen) ? k : (i + k - maxLen))) % maxLen] - maxLog);
					}
				}
			}
			if (maxLog != log2(0)) {
				buffer[maxLen * j + (queueStart + ((i > maxLen) ? maxLen : i)) % maxLen] = maxLog + log2(sumLog);
			}
		}
		maxLog = log2(0);
		checksum[(queueStart + ((i > maxLen) ? maxLen : i)) % maxLen] = 0;
		buffer[(queueStart + ((i > maxLen) ? maxLen : i)) % maxLen] = log2(0);
		for (size_t j = 0; j < maxLen; j++) {
			if (maxLen <= i + j  && scratch[j]) {
				if (wrappedString[i + j - maxLen].valid) {
					checksum[(queueStart + ((i > maxLen) ? maxLen : i)) % maxLen] = 1;
				}
				if (wrappedString[i + j - maxLen].sampleLog > maxLog) {
					maxLog = wrappedString[i + j - maxLen].sampleLog;
				}
			}
		}
		sumLog = 0;
		for (size_t j = 0; j < maxLen; j++) {
			if (maxLen <= i + j && scratch[j]) {
				if (maxLog != log2(0)) {
					sumLog += pow(2, wrappedString[i + j - maxLen].sampleLog - maxLog);
				}
			}
		}
		if (maxLog != log2(0)) {
			buffer[(queueStart + ((i > maxLen) ? maxLen : i)) % maxLen] = maxLog + log2(sumLog);
		}
		for (size_t j = 0; j < maxLen; j++) {
			scratch[j] = 0;
		}
	}
	struct node *lastVocab = NULL;
	char atomicExists = 0;
	for (struct node *i = *vocab; i != NULL; i = i->next) {
		lastVocab = i;
		if (!strcmp(wrappedString[numCodepoints - 1].content, i->vocab)) {
			atomicExists = 1;
		}
	}
	if (!atomicExists) {
		struct node *temp = malloc(sizeof(struct node));
		if (temp == NULL) {
			free(buffer);
			free(checksum);
			free(scratch);
			return -1;
		}
		temp->vocab = malloc((strlen(wrappedString[numCodepoints - 1].content) + 1) * sizeof(char));
		if (temp->vocab == NULL) {
			free(buffer);
			free(checksum);
			free(scratch);
			free(temp);
			return -1;
		}
		strcpy(temp->vocab, wrappedString[numCodepoints - 1].content);
		temp->next = NULL;
		if (lastVocab == NULL) {
			*vocab = temp;
		} else {
			lastVocab->next = temp;
			long int newVocPos = ftell(vocabFile);
			if (newVocPos < 0) {
				free(buffer);
				free(scratch);
				free(checksum);
				return -1;
			}
			if (newVocPos != 0) {
				if (fwrite(",", sizeof(char), 1, vocabFile) != 1) {
					free(buffer);
					free(scratch);
					free(checksum);
					return -1;
				}
			}
		}
		if (!strcmp(wrappedString[numCodepoints - 1].content, "\"")) {
			if (fwrite("\"\"\"\"", sizeof(char), 4, vocabFile) != 4) {
				free(buffer);
				free(scratch);
				free(checksum);
				return -1;
			}
		} else if (!strcmp(wrappedString[numCodepoints - 1].content, ",")) {
			if (fwrite("\",\"", sizeof(char), 3, vocabFile) != 3) {
				free(buffer);
				free(scratch);
				free(checksum);
				return -1;
			}
		} else if (!strcmp(wrappedString[numCodepoints - 1].content, "\n")) {
			if (fwrite("\"\n\"", sizeof(char), 3, vocabFile) != 3) {
				free(buffer);
				free(scratch);
				free(checksum);
				return -1;
			}
		} else {
			if (fwrite(wrappedString[numCodepoints - 1].content, sizeof(char), strlen(wrappedString[numCodepoints - 1].content), vocabFile) != strlen(wrappedString[numCodepoints - 1].content)) {
				free(buffer);
				free(scratch);
				free(checksum);
				return -1;
			}
		}
	}
	for (size_t i = 0; i < maxLen * (context - 1); i++) {
		buffer[i] = 0;
		checksum[i] = 0;
	}
	for (size_t i = numCodepoints; i > 0; i--) {
		i--;
		for (struct node *j = *vocab; j != NULL; j = j->next) {
			size_t compDist = 0;
			char match = 1;
			size_t k = i;
			for (; k < numCodepoints && compDist + strlen(wrappedString[k].content) <= strlen(j->vocab) && match; k++) {
				char temp = j->vocab[compDist + strlen(wrappedString[k].content)];
				j->vocab[compDist + strlen(wrappedString[k].content)] = '\0';
				if (strcmp(j->vocab + compDist, wrappedString[k].content)) {
					match = 0;
				}
				j->vocab[compDist + strlen(wrappedString[k].content)] = temp;
				compDist += strlen(wrappedString[k].content);
			}
			if (compDist != strlen(j->vocab)) {
				match = 0;
			}
			if (match) {
				scratch[k - i - 1] = 1;
			}
		}
		size_t minLength = numCodepoints;
		double maxLog = log2(0);
		for (size_t j = 0; j < maxLen; j++) {
			if (scratch[j]) {
				if (i + j + 1 == numCodepoints) {
					minLength = 0;
					if (maxLog < log2(1)) {
						maxLog = log2(1);
					}
				} else {
					if (minLength > wrappedString[i + j + 1].numTokens) {
						minLength = wrappedString[i + j + 1].numTokens;
					}
					if (maxLog < wrappedString[i + j + 1].logTokenizations) {
						maxLog = wrappedString[i + j + 1].logTokenizations;
					}
				}
			}
		}
		wrappedString[i].numTokens = minLength + 1;
		double sumLog = 0;
		for (size_t j = 0; j < maxLen; j++) {
			if (scratch[j]) {
				if (i + j + 1 == numCodepoints) {
					if (maxLog != log2(0)) {
						sumLog += pow(2, log2(1) - maxLog);
					}
				} else {
					if (maxLog != log2(0)) {
						sumLog += pow(2, wrappedString[i + j + 1].logTokenizations - maxLog);
					}
				}
			}
		}
		if (maxLog != log2(0)) {
			wrappedString[i].logTokenizations = maxLog + log2(sumLog);
		}
		if (wrappedString[i].valid) {
			for (size_t j = 0; j < maxLen; j++) {
				if (scratch[j] && (i + j + 1 == numCodepoints || checksum[(queueStart + j + 1) % maxLen])) {
					wrappedString[i].entropy += (i + j + 1 == numCodepoints) ? (pow(2, -wrappedString[i].logTokenizations) * wrappedString[i].logTokenizations) : (pow(2, wrappedString[i + j + 1].logTokenizations - wrappedString[i].logTokenizations) * (-wrappedString[i + j + 1].logTokenizations + wrappedString[i].logTokenizations + buffer[(queueStart + j + 1) % maxLen]));
				}
			}
		}
		for (size_t j = 0; j < context - 2; j++) {
			checksum[maxLen * j + queueStart] = 0;
			buffer[maxLen * j + queueStart] = 0;
			for (size_t k = 0; k < maxLen; k++) {
				if (scratch[k] && (i + k + 1 == numCodepoints || checksum[maxLen * (j + 1) + (queueStart + k + 1) % maxLen])) {
					checksum[maxLen * j + queueStart] = 1;
					buffer[maxLen * j + queueStart] += (i + k + 1 == numCodepoints) ? (pow(2, -wrappedString[i].logTokenizations) * wrappedString[i].logTokenizations) : (pow(2, wrappedString[i + k + 1].logTokenizations - wrappedString[i].logTokenizations) * (-wrappedString[i + k + 1].logTokenizations + wrappedString[i].logTokenizations + buffer[maxLen * (j + 1) + (queueStart + k + 1) % maxLen]));
				}
			}
		}
		checksum[maxLen * (context - 2) + queueStart] = 0;
		buffer[maxLen * (context - 2) + queueStart] = 0;
		for (size_t j = 0; j < maxLen; j++) {
			if (scratch[j] && (i + j + 1 == numCodepoints || wrappedString[i + j + 1].valid)) {
				checksum[maxLen * (context - 2) + queueStart] = 1;
				buffer[maxLen * (context - 2) + queueStart] += (i + j + 1 == numCodepoints) ? (pow(2, -wrappedString[i].logTokenizations) * wrappedString[i].logTokenizations) : (pow(2, wrappedString[i + j + 1].logTokenizations - wrappedString[i].logTokenizations) * (-wrappedString[i + j + 1].logTokenizations + wrappedString[i].logTokenizations));
			}
		}
		for (size_t j = 0; j < maxLen; j++) {
			scratch[j] = 0;
		}
		queueStart = (queueStart == 0) ? (maxLen - 1) : (queueStart - 1);
		i++;
	}
	for (size_t i = 0; i < numCodepoints; i++) {
		if (wrappedString[i].valid) {
			for (size_t j = 0; j < wrappedString[i].entropy / log2(exp(1.0)) * scaling; j++) {
				if (pow(2, wrappedString[i].sampleLog + wrappedString[i].logTokenizations - wrappedString[0].logTokenizations) * ((unsigned int)RAND_MAX + 1u) > rand()) {
					size_t vPos = i;
					for (size_t k = 0; k < context && vPos < numCodepoints - 1; k++) {
						double random = (double)rand() / ((unsigned int)RAND_MAX + 1u);
						size_t ind = 0;
						size_t lastValid;
						size_t lastInd;
						for (struct node *l = *vocab; random >= 0 && l != NULL; l = l->next) {
							size_t compDist = 0;
							char match = 1;
							size_t m = vPos;
							for (; m < numCodepoints && compDist + strlen(wrappedString[m].content) <= strlen(l->vocab) && match; m++) {
								char temp = l->vocab[compDist + strlen(wrappedString[m].content)];
								l->vocab[compDist + strlen(wrappedString[m].content)] = '\0';
								if (strcmp(l->vocab + compDist, wrappedString[m].content)) {
									match = 0;
								}
								l->vocab[compDist + strlen(wrappedString[m].content)] = temp;
								compDist += strlen(wrappedString[m].content);
							}
							if (compDist != strlen(l->vocab)) {
								match = 0;
							}
							if (match) {
								if (m == numCodepoints) {
									random -= pow(2, -wrappedString[vPos].logTokenizations);
								} else {
									random -= pow(2, wrappedString[m].logTokenizations - wrappedString[vPos].logTokenizations);
								}
								lastValid = m;
								lastInd = ind;
							}
							if (random >= 0 && l->next != NULL) {
								ind++;
							} else {
								vPos = lastValid;
							}
						}
						ind = lastInd;
						if (vPos < numCodepoints) {
							if (!k && *startPoint > 0) {
								if (fwrite("\n", sizeof(char), 1, dump) != 1) {
									free(buffer);
									free(checksum);
									free(scratch);
									return -1;
								}
								(*startPoint)++;
							}
							if (k) {
								if (fwrite(",", sizeof(char), 1, dump) != 1) {
									free(buffer);
									free(checksum);
									free(scratch);
									return -1;
								}
								(*startPoint)++;
							}
							if (fwrite("\"", sizeof(char), 1, dump) != 1) {
								free(buffer);
								free(checksum);
								free(scratch);
								return -1;
							}
							(*startPoint)++;
							int numWritten;
							if ((numWritten = fprintf(dump, "%zu", ind)) < 0) {
								free(buffer);
								free(checksum);
								free(scratch);
								return -1;
							}
							*startPoint += numWritten;
							if (fwrite(",", sizeof(char), 1, dump) != 1) {
								free(buffer);
								free(checksum);
								free(scratch);
								return -1;
							}
							(*startPoint)++;
							size_t maxVocab = 0;
							ind = 0;
							size_t maxInd = 0;
							for (struct node *l = *vocab; l != NULL; l = l->next) {
								size_t compDist = 0;
								char match = 1;
								size_t m = vPos;
								for (; m < numCodepoints && compDist + strlen(wrappedString[m].content) <= strlen(l->vocab) && match; m++) {
									char temp = l->vocab[compDist + strlen(wrappedString[m].content)];
									l->vocab[compDist + strlen(wrappedString[m].content)] = '\0';
									if (strcmp(l->vocab + compDist, wrappedString[m].content)) {
										match = 0;
									}
									l->vocab[compDist + strlen(wrappedString[m].content)] = temp;
									compDist += strlen(wrappedString[m].content);
								}
								if (compDist != strlen(l->vocab)) {
									match = 0;
								}
								if (match) {
									if (m == numCodepoints) {
										if (wrappedString[vPos].numTokens == 1 && strlen(l->vocab) > maxVocab) {
											maxVocab = strlen(l->vocab);
											maxInd = ind;
										}
									} else {
										if (wrappedString[vPos].numTokens == wrappedString[m].numTokens + 1 && strlen(l->vocab) > maxVocab) {
											maxVocab = strlen(l->vocab);
											maxInd = ind;
										}
									}
								}
								ind++;
							}
							if ((numWritten = fprintf(dump, "%zu", maxInd)) < 0) {
								free(buffer);
								free(checksum);
								free(scratch);
								return -1;
							}
							*startPoint += numWritten;
							if (fwrite("\"", sizeof(char), 1, dump) != 1) {
								free(buffer);
								free(checksum);
								free(scratch);
								return -1;
							}
							(*startPoint)++;
						}
					}
				}
			}
		}
	}
	free(buffer);
	free(checksum);
	free(scratch);
	return 0;
}
struct node *getVocab(FILE *vocabFile) {
	struct node *firstVocab = malloc(sizeof(struct node));
	if (firstVocab == NULL) {
		return NULL;
	}
	firstVocab->next = NULL;
	struct node *curVocab;
	char escaped = 1;
	char start = 0;
	char quoted = 0;
	char mode = 0;
	char first = 0;
	long int dist = 0;
	long int vocLen = 0;
	char fileStart = 1;
	while (!feof(vocabFile)) {
		int ch = getc(vocabFile);
		if (ch == EOF && !feof(vocabFile)) {
			for (struct node *i = firstVocab->next; i != NULL;) {
				free(i->vocab);
				struct node *j = i->next;
				free(i);
				i = j;
			}
			free(firstVocab);
			return NULL;
		}
		if (fileStart && !feof(vocabFile) && ch == '"') {
			start = 1;
			fileStart = 0;
		} else if (fileStart && !feof(vocabFile)) {
			fileStart = 0;
		}
		if (!feof(vocabFile) && !mode) {
			dist--;
		}
		if (start) {
			start = 0;
			if (!feof(vocabFile) && ch == '"') {
				escaped = 0;
				quoted = 1;
				first = 1;
			}
		}
		if ((feof(vocabFile) && !fileStart) || ((ch == ',' || ch == '\n') && escaped)) {
			if (mode) {
				mode = 0;
			} else {
				if (fseek(vocabFile, dist, SEEK_CUR)) {
					for (struct node *i = firstVocab->next; i != NULL;) {
						free(i->vocab);
						struct node *j = i->next;
						free(i);
						i = j;
					}
					free(firstVocab);
					return NULL;
				}
				dist = 0;
				mode = 1;
				struct node *nextVocab = malloc(sizeof(struct node));
				if (nextVocab == NULL) {
					for (struct node *i = firstVocab->next; i != NULL;) {
						free(i->vocab);
						struct node *j = i->next;
						free(i);
						i = j;
					}
					free(firstVocab);
					return NULL;
				}
				nextVocab->vocab = malloc((vocLen + 1) * sizeof(char));
				if (nextVocab->vocab == NULL) {
					for (struct node *i = firstVocab->next; i != NULL;) {
						free(i->vocab);
						struct node *j = i->next;
						free(i);
						i = j;
					}
					free(firstVocab);
					free(nextVocab);
					return NULL;
				}
				nextVocab->vocab[vocLen] = '\0';
				nextVocab->next = NULL;
				if (firstVocab->next == NULL) {
					firstVocab->next = nextVocab;
				} else {
					curVocab->next = nextVocab;
				}
				curVocab = nextVocab;
			}
			start = 1;
			quoted = 0;
			vocLen = 0;
		} else if (!feof(vocabFile)) {
			if (first) {
				first = 0;
			} else if (quoted) {
				if (escaped) {
					if (ch == '"') {
						escaped = 0;
						if (mode) {
							curVocab->vocab[vocLen] = ch;
						}
						vocLen++;
					} else {
						for (struct node *i = firstVocab->next; i != NULL;) {
							free(i->vocab);
							struct node *j = i->next;
							free(i);
							i = j;
						}
						free(firstVocab);
						return NULL;
					}
				} else {
					if (ch == '"') {
						escaped = 1;
					} else {
						if (mode) {
							curVocab->vocab[vocLen] = ch;
						}
						vocLen++;
					}
				}
			} else {
				if (ch == '"') {
					for (struct node *i = firstVocab->next; i != NULL;) {
						free(i->vocab);
						struct node *j = i->next;
						free(i);
						i = j;
					}
					free(firstVocab);
					return NULL;
				}
				if (mode) {
					curVocab->vocab[vocLen] = ch;
				}
				vocLen++;
			}
		}
	}
	return firstVocab;
}
int main(int argc, char *argv[]) {
	(void)argc;
	unsigned seed = 5381;
	for (size_t i = 0; argv[3][i] != '\0'; i++) {
		seed = (seed << 5) + seed + argv[3][i];
	}
	srand(time(NULL) + seed);
	FILE *dump = fopen(argv[1], "wb+");
	if (dump == NULL) {
		return 1;
	}
	FILE *vocabFile = fopen(argv[2], "rb+");
	if (vocabFile == NULL) {
		fclose(dump);
		return 1;
	}
	struct node *vocab = getVocab(vocabFile);
	if (vocab == NULL) {
		fclose(vocabFile);
		fclose(dump);
		return 1;
	}
	struct node *nextVocab = vocab->next;
	free(vocab);
	vocab = nextVocab;
	size_t maxLen = 0;
	for (struct node *i = vocab; i != NULL; i = i->next) {
		int vocLen = 0;
		char vocTot = 0;
		size_t compLen = 0;
		char vocBuf[4];
		for (size_t j = 0; i->vocab[j] != '\0'; j++) {
			if (vocLen < vocTot && (i->vocab[j] & 192) == 128) {
				vocBuf[vocLen] = i->vocab[j];
				vocLen++;
			} else {
				size_t utfValue = vocBuf[0] & ((1 << (7 - vocTot)) - 1);
				if (vocTot > 0 && vocLen == vocTot) {
					for (int k = 1; k < vocTot; k++) {
						utfValue = (utfValue << 6) + (vocBuf[k] & 63);
					}
				}
				if (vocTot > 0 && vocLen == vocTot && utfValue > 127 && utfValue < 1114112 && (((utfValue >> 7) && vocTot == 2) || ((utfValue >> 11) && vocTot == 3) || ((utfValue >> 16) && vocTot == 4))) {
					compLen++;
				} else if (vocTot) {
					compLen += vocLen;
				}
				vocLen = 0;
				vocTot = 0;
				if ((i->vocab[j] & 224) == 192) {
					vocTot = 2;
					vocLen = 1;
					vocBuf[0] = i->vocab[j];
				} else if ((i->vocab[j] & 240) == 224) {
					vocTot = 3;
					vocLen = 1;
					vocBuf[0] = i->vocab[j];
	 			} else if ((i->vocab[j] & 248) == 240) {
					vocTot = 4;
					vocLen = 1;
					vocBuf[0] = i->vocab[j];
				} else {
					compLen++;
				}
			}
		}
		size_t utfValue = vocBuf[0] & ((1 << (7 - vocTot)) - 1);
		if (vocTot > 0 && vocLen == vocTot) {
			for (int k = 1; k < vocTot; k++) {
				utfValue = (utfValue << 6) + (vocBuf[k] & 63);
			}
		}
		if (vocLen < vocTot || utfValue < 128 || utfValue > 1114111 || !(((utfValue >> 7) && vocTot == 2) || ((utfValue >> 11) && vocTot == 3) || ((utfValue >> 16) && vocTot == 4))) {
			compLen += vocLen;
		} else if (vocTot) {
			compLen++;
		}
		if (compLen > maxLen) {
			maxLen = compLen;
		}
	}
	if (maxLen < 1) {
		maxLen = 1;
	}
	if (fclose(vocabFile) == EOF) {
		for (struct node *i = vocab; i != NULL;) {
			free(i->vocab);
			struct node *j = i->next;
			free(i);
			i = j;
		}
		fclose(dump);
		return 1;
	}
	vocabFile = fopen(argv[3], "wb+");
	if (vocabFile == NULL) {
		for (struct node *i = vocab; i != NULL;) {
			free(i->vocab);
			struct node *j = i->next;
			free(i);
			i = j;
		}
		fclose(dump);
		return 1;
	}
	char buffer[4];
	int bufLen = 0;
	int totLen = 0;
	long int startPoint = 0;
	size_t numCodepoints = 0;
	char start = 1;
	while (!feof(stdin)) {
		int ch = getc(stdin);
		if (ch == EOF && !feof(stdin)) {
			for (struct node *i = vocab; i != NULL;) {
				free(i->vocab);
				struct node *j = i->next;
				free(i);
				i = j;
			}
			fclose(vocabFile);
			fclose(dump);
			return 1;
		}
		if (ch == '\0' || (feof(stdin) && !start)) {
			start = 1;
			size_t utfValue = buffer[0] & ((1 << (7 - totLen)) - 1);
			if (totLen > 0 && bufLen == totLen) {
				for (int i = 1; i < totLen; i++) {
					utfValue = (utfValue << 6) + (buffer[i] & 63);
				}
			}
			if (bufLen < totLen || utfValue < 128 || utfValue > 1114111 || !(((utfValue >> 7) && totLen == 2) || ((utfValue >> 11) && totLen == 3) || ((utfValue >> 16) && totLen == 4))) {
				for (char i = 0; i < bufLen; i++) {
					struct codepoint temp;
					temp.numTokens = 0;
					temp.sampleLog = log2(0);
					temp.logTokenizations = log2(0);
					temp.entropy = 0;
					temp.valid = 0;
					temp.content[0] = *(buffer + i);
					temp.content[1] = '\0';
					if (fwrite(&temp, sizeof(struct codepoint), 1, dump) != 1) {
						for (struct node *i = vocab; i != NULL;) {
							free(i->vocab);
							struct node *j = i->next;
							free(i);
							i = j;
						}
						fclose(dump);
						fclose(vocabFile);
						return 1;
					}
				}
				numCodepoints += bufLen;
			} else if (totLen) {
				struct codepoint temp;
				temp.numTokens = 0;
				temp.sampleLog = log2(0);
				temp.logTokenizations = log2(0);
				temp.entropy = 0;
				temp.valid = 0;
				for (int i = 0; i < bufLen; i++) {
					temp.content[i] = *(buffer + i);
				}
				temp.content[bufLen] = '\0';
				if (fwrite(&temp, sizeof(struct codepoint), 1, dump) != 1) {
					for (struct node *i = vocab; i != NULL;) {
						free(i->vocab);
						struct node *j = i->next;
						free(i);
						i = j;
					}
					fclose(vocabFile);
					fclose(dump);
					return 1;
				}
				numCodepoints++;
			}
			bufLen = 0;
			totLen = 0;
			struct codepoint *wrappedString = malloc(numCodepoints * sizeof(struct codepoint));
			if (wrappedString == NULL) {
				for (struct node *i = vocab; i != NULL;) {
					free(i->vocab);
					struct node *j = i->next;
					free(i);
					i = j;
				}
				fclose(vocabFile);
				fclose(dump);
				return 1;
			}
			if (fseek(dump, -1 * numCodepoints * sizeof(struct codepoint), SEEK_CUR) < 0) {
				free(wrappedString);
				for (struct node *i = vocab; i != NULL;) {
					free(i->vocab);
					struct node *j = i->next;
					free(i);
					i = j;
				}
				fclose(vocabFile);
				fclose(dump);
				return 1;
			}
			if (fread(wrappedString, sizeof(struct codepoint), numCodepoints, dump) != numCodepoints) {
				free(wrappedString);
				for (struct node *i = vocab; i != NULL;) {
					free(i->vocab);
					struct node *j = i->next;
					free(i);
					i = j;
				}
				fclose(dump);
				fclose(vocabFile);
				return 1;
			}
			if (fseek(dump, -1 * numCodepoints * sizeof(struct codepoint), SEEK_CUR) < 0) {
				free(wrappedString);
				for (struct node *i = vocab; i != NULL;) {
					free(i->vocab);
					struct node *j = i->next;
					free(i);
					i = j;
				}
				fclose(vocabFile);
				fclose(dump);
				return 1;
			}
			if (generateSamplesFromString(dump, vocabFile, maxLen, wrappedString, numCodepoints, &vocab, atoi(argv[4]), strtod(argv[5], NULL), &startPoint) == -1) {
				free(wrappedString);
				for (struct node *i = vocab; i != NULL;) {
					free(i->vocab);
					struct node *j = i->next;
					free(i);
					i = j;
				}
				fclose(vocabFile);
				fclose(dump);
				return 1;
			}
			numCodepoints = 0;
			free(wrappedString);
		} else if (!feof(stdin)) {
			start = 0;
			if (bufLen < totLen && (ch & 192) == 128) {
				buffer[bufLen] = ch;
				bufLen++;
			} else {
				size_t utfValue = buffer[0] & ((1 << (7 - totLen)) - 1);
				if (totLen > 0 && bufLen == totLen) {
					for (int i = 1; i < totLen; i++) {
						utfValue = (utfValue << 6) + (buffer[i] & 63);
					}
				}
				if (totLen > 0 && bufLen == totLen && utfValue > 127 && utfValue < 1114112 && (((utfValue >> 7) && totLen == 2) || ((utfValue >> 11) && totLen == 3) || ((utfValue >> 16) && totLen == 4))) {
					struct codepoint temp;
					temp.numTokens = 0;
					temp.sampleLog = log2(0);
					temp.logTokenizations = log2(0);
					temp.entropy = 0;
					temp.valid = 0;
					for (int i = 0; i < totLen; i++) {
						temp.content[i] = *(buffer + i);
					}
					temp.content[totLen] = '\0';
					if (fwrite(&temp, sizeof(struct codepoint), 1, dump) != 1) {
						for (struct node *i = vocab; i != NULL;) {
							free(i->vocab);
							struct node *j = i->next;
							free(i);
							i = j;
						}
						fclose(vocabFile);
						fclose(dump);
						return 1;
					}
					numCodepoints++;
				} else if (totLen) {
					for (char i = 0; i < bufLen; i++) {
						struct codepoint temp;
						temp.numTokens = 0;
						temp.sampleLog = log2(0);
						temp.logTokenizations = log2(0);
						temp.entropy = 0;
						temp.valid = 0;
						temp.content[0] = *(buffer + i);
						temp.content[1] = '\0';
						if (fwrite(&temp, sizeof(struct codepoint), 1, dump) != 1) {
							for (struct node *i = vocab; i != NULL;) {
								free(i->vocab);
								struct node *j = i->next;
								free(i);
								i = j;
							}
							fclose(vocabFile);
							fclose(dump);
							return 1;
						}
					}
					numCodepoints += bufLen;
				}
				bufLen = 0;
				totLen = 0;
				if ((ch & 224) == 192) {
					totLen = 2;
					buffer[0] = ch;
					bufLen = 1;
				} else if ((ch & 240) == 224) {
					totLen = 3;
					buffer[0] = ch;
					bufLen = 1;
	 			} else if ((ch & 248) == 240) {
					totLen = 4;
					buffer[0] = ch;
					bufLen = 1;
				} else {
					struct codepoint temp;
					temp.numTokens = 0;
					temp.sampleLog = log2(0);
					temp.logTokenizations = log2(0);
					temp.entropy = 0;
					temp.valid = 0;
					temp.content[0] = ch;
					temp.content[1] = '\0';
					if (fwrite(&temp, sizeof(struct codepoint), 1, dump) != 1) {
						for (struct node *i = vocab; i != NULL;) {
							free(i->vocab);
							struct node *j = i->next;
							free(i);
							i = j;
						}
						fclose(vocabFile);
						fclose(dump);
						return 1;
					}
					numCodepoints++;
				}
			}
		}
	}
	if (printf("%ld", startPoint) < 0) {
		for (struct node *i = vocab; i != NULL;) {
			free(i->vocab);
			struct node *j = i->next;
			free(i);
			i = j;
		}
		fclose(vocabFile);
		fclose(dump);
		return 1;
	}
	for (struct node *i = vocab; i != NULL;) {
		free(i->vocab);
		struct node *j = i->next;
		free(i);
		i = j;
	}
	if (fclose(vocabFile) == EOF) {
		fclose(dump);
		return 1;
	}
	if (fclose(dump) == EOF) {
		return 1;
	}
	return 0;
}