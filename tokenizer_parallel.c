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
	struct node* next[256];
	struct node* backtrack;
	size_t vocab;
	char leaf;
};
long int generateSamplesFromString(FILE *dump, FILE *vocabFile, size_t *numVocabs, size_t maxLen, struct codepoint *wrappedString, size_t numCodepoints, struct node **vocab, size_t context, double scaling, long int *startPoint) {
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
		struct node **lastVocab = vocab;
		struct codepoint curPoint = wrappedString[i - 1];
		char codeIndex = strlen(curPoint.content) - 1;
		size_t traveled = 1;
		while (i >= traveled && maxLen >= traveled && lastVocab[(unsigned char)curPoint.content[codeIndex]] != NULL) {
			if (!codeIndex) {
				scratch[maxLen - traveled] = lastVocab[(unsigned char)curPoint.content[codeIndex]]->leaf;
				if (traveled == 1 && scratch[maxLen - traveled]) {
					atomicExists = 1;
				}
				traveled++;
				if (i >= traveled) {
					lastVocab = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next;
					curPoint = wrappedString[i - traveled];
					codeIndex = strlen(curPoint.content) - 1;
				}
			} else {
				if (i >= traveled) {
					lastVocab = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next;
				}
				codeIndex--;
			}
		}
		if (!atomicExists) {
			struct node **vocabs = vocab;
			struct node *curNode = NULL;
			for (char j = strlen(wrappedString[i - 1].content); j > 0; j--) {
				if (curNode == NULL) {
					if (vocabs[(unsigned char)wrappedString[i - 1].content[j - 1]] == NULL) {
						vocabs[(unsigned char)wrappedString[i - 1].content[j - 1]] = malloc(sizeof(struct node));
						if (vocabs[(unsigned char)wrappedString[i - 1].content[j - 1]] == NULL) {
							free(buffer);
							free(scratch);
							free(checksum);
							return -1;
						}
						vocabs[(unsigned char)wrappedString[i - 1].content[j - 1]]->leaf = 0;
						vocabs[(unsigned char)wrappedString[i - 1].content[j - 1]]->backtrack = NULL;
						for (int k = 0; k < 256; k++) {
							vocabs[(unsigned char)wrappedString[i - 1].content[j - 1]]->next[k] = NULL;
						}
					}
					curNode = vocabs[(unsigned char)wrappedString[i - 1].content[j - 1]];
				} else {
					if (curNode->next[(unsigned char)wrappedString[i - 1].content[j - 1]] == NULL) {
						curNode->next[(unsigned char)wrappedString[i - 1].content[j - 1]] = malloc(sizeof(struct node));
						if (curNode->next[(unsigned char)wrappedString[i - 1].content[j - 1]] == NULL) {
							free(buffer);
							free(scratch);
							free(checksum);
							return -1;
						}
						curNode->next[(unsigned char)wrappedString[i - 1].content[j - 1]]->leaf = 0;
						curNode->next[(unsigned char)wrappedString[i - 1].content[j - 1]]->backtrack = curNode;
						for (int k = 0; k < 256; k++) {
							curNode->next[(unsigned char)wrappedString[i - 1].content[j - 1]]->next[k] = NULL;
						}
					}
					curNode = curNode->next[(unsigned char)wrappedString[i - 1].content[j - 1]];
				}
			}
			curNode->leaf = 1;
			curNode->vocab = *numVocabs;
			vocabs = vocab;
			curNode = NULL;
			for (char j = 0; j < strlen(wrappedString[i - 1].content) + 1; j++) {
				if (curNode == NULL) {
					if (vocabs[(unsigned char)wrappedString[i - 1].content[j]] == NULL) {
						vocabs[(unsigned char)wrappedString[i - 1].content[j]] = malloc(sizeof(struct node));
						if (vocabs[(unsigned char)wrappedString[i - 1].content[j]] == NULL) {
							free(buffer);
							free(scratch);
							free(checksum);
							return -1;
						}
						vocabs[(unsigned char)wrappedString[i - 1].content[j]]->leaf = 0;
						vocabs[(unsigned char)wrappedString[i - 1].content[j]]->backtrack = NULL;
						for (int k = 0; k < 256; k++) {
							vocabs[(unsigned char)wrappedString[i - 1].content[j]]->next[k] = NULL;
						}
					}
					curNode = vocabs[(unsigned char)wrappedString[i - 1].content[j]];
				} else {
					if (curNode->next[(unsigned char)wrappedString[i - 1].content[j]] == NULL) {
						curNode->next[(unsigned char)wrappedString[i - 1].content[j]] = malloc(sizeof(struct node));
						if (curNode->next[(unsigned char)wrappedString[i - 1].content[j]] == NULL) {
							free(buffer);
							free(scratch);
							free(checksum);
							return -1;
						}
						curNode->next[(unsigned char)wrappedString[i - 1].content[j]]->leaf = 0;
						curNode->next[(unsigned char)wrappedString[i - 1].content[j]]->backtrack = curNode;
						for (int k = 0; k < 256; k++) {
							curNode->next[(unsigned char)wrappedString[i - 1].content[j]]->next[k] = NULL;
						}
					}
					curNode = curNode->next[(unsigned char)wrappedString[i - 1].content[j]];
				}
			}
			curNode->leaf = 1;
			curNode->vocab = *numVocabs;
			(*numVocabs)++;
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
	struct node **lastVocab = vocab;
	char atomicExists = 0;
	char ii = 0;
	for (; wrappedString[numCodepoints - 1].content[ii] != '\0' && lastVocab[(unsigned char)wrappedString[numCodepoints - 1].content[ii]] != NULL; ii++) {
		lastVocab = lastVocab[(unsigned char)wrappedString[numCodepoints - 1].content[ii]]->next;
	}
	atomicExists = (wrappedString[numCodepoints - 1].content[ii] != '\0') && (lastVocab[0] != NULL);
	if (!atomicExists) {
		struct node **vocabs = vocab;
		struct node *curNode = NULL;
		for (char j = strlen(wrappedString[numCodepoints - 1].content); j > 0; j--) {
			if (curNode == NULL) {
				if (vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]] == NULL) {
					vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]] = malloc(sizeof(struct node));
					if (vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]] == NULL) {
						free(buffer);
						free(scratch);
						free(checksum);
						return -1;
					}
					vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]]->leaf = 0;
					vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]]->backtrack = NULL;
					for (int k = 0; k < 256; k++) {
						vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]]->next[k] = NULL;
					}
				}
				curNode = vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]];
			} else {
				if (curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]] == NULL) {
					curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]] = malloc(sizeof(struct node));
					if (curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]] == NULL) {
						free(buffer);
						free(scratch);
						free(checksum);
						return -1;
					}
					curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]]->leaf = 0;
					curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]]->backtrack = curNode;
					for (int k = 0; k < 256; k++) {
						curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]]->next[k] = NULL;
					}
				}
				curNode = curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j - 1]];
			}
		}
		curNode->leaf = 1;
		curNode->vocab = *numVocabs;
		vocabs = vocab;
		curNode = NULL;
		for (char j = 0; j < strlen(wrappedString[numCodepoints - 1].content) + 1; j++) {
			if (curNode == NULL) {
				if (vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j]] == NULL) {
					vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j]] = malloc(sizeof(struct node));
					if (vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j]] == NULL) {
						free(buffer);
						free(scratch);
						free(checksum);
						return -1;
					}
					vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j]]->leaf = 0;
					vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j]]->backtrack = NULL;
					for (int k = 0; k < 256; k++) {
						vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j]]->next[k] = NULL;
					}
				}
				curNode = vocabs[(unsigned char)wrappedString[numCodepoints - 1].content[j]];
			} else {
				if (curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j]] == NULL) {
					curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j]] = malloc(sizeof(struct node));
					if (curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j]] == NULL) {
						free(buffer);
						free(scratch);
						free(checksum);
						return -1;
					}
					curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j]]->leaf = 0;
					curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j]]->backtrack = curNode;
					for (int k = 0; k < 256; k++) {
						curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j]]->next[k] = NULL;
					}
				}
				curNode = curNode->next[(unsigned char)wrappedString[numCodepoints - 1].content[j]];
			}
		}
		curNode->leaf = 1;
		curNode->vocab = *numVocabs;
		(*numVocabs)++;
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
	free(checksum);
	double *leftovers = malloc(maxLen * (context - 1) * sizeof(double));
	if (leftovers == NULL) {
		free(buffer);
		free(scratch);
		return -1;
	}
	for (size_t i = 0; i < maxLen * (context - 1); i++) {
		buffer[i] = 0;
	}
	for (size_t i = numCodepoints; i > 0; i--) {
		i--;
		lastVocab = vocab;
		struct codepoint curPoint = wrappedString[i];
		char codeIndex = 0;
		size_t traveled = 0;
		while (i + traveled < numCodepoints && traveled < maxLen && lastVocab[(unsigned char)curPoint.content[codeIndex]] != NULL) {
			if (codeIndex == strlen(curPoint.content) - 1) {
				scratch[traveled] = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next[0] != NULL;
				if (!traveled && scratch[traveled]) {
					atomicExists = 1;
				}
				traveled++;
				if (i + traveled < numCodepoints) {
					lastVocab = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next;
					curPoint = wrappedString[i + traveled];
				}
				codeIndex = 0;
			} else {
				if (i + traveled < numCodepoints) {
					lastVocab = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next;
				}
				codeIndex++;
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
				if (scratch[j]) {
					wrappedString[i].entropy += (i + j + 1 == numCodepoints) ? (pow(2, -wrappedString[i].logTokenizations) * wrappedString[i].logTokenizations) : (pow(2, wrappedString[i + j + 1].logTokenizations - wrappedString[i].logTokenizations) * (-wrappedString[i + j + 1].logTokenizations + wrappedString[i].logTokenizations + buffer[(queueStart + j + 1) % maxLen]));
				}
			}
		}
		for (size_t j = 0; j < context - 2; j++) {
			buffer[maxLen * j + queueStart] = 0;
			leftovers[maxLen * j + queueStart] = 0;
			double maxLeft = log2(0);
			for (size_t k = 0; k < maxLen; k++) {
				if (scratch[k]) {
					buffer[maxLen * j + queueStart] += (i + k + 1 == numCodepoints) ? (pow(2, -wrappedString[i].logTokenizations) * wrappedString[i].logTokenizations) : (pow(2, wrappedString[i + k + 1].logTokenizations - wrappedString[i].logTokenizations) * (-wrappedString[i + k + 1].logTokenizations + wrappedString[i].logTokenizations + buffer[maxLen * (j + 1) + (queueStart + k + 1) % maxLen]));
					if (i + k + 1 < numCodepoints && maxLeft < leftovers[maxLen * (j + 1) + (queueStart + k + 1) % maxLen]) {
						maxLeft = leftovers[maxLen * (j + 1) + (queueStart + k + 1) % maxLen];
					}
				}
			}
			for (size_t k = 0; k < maxLen; k++) {
				if (maxLeft != log2(0) && i + k + 1 < numCodepoints && scratch[k]) {
					leftovers[maxLen * j + queueStart] += pow(2, leftovers[maxLen * (j + 1) + (queueStart + k + 1) % maxLen] - maxLeft);
				}
			}
			if (leftovers[maxLen * j + queueStart]) {
				leftovers[maxLen * j + queueStart] = maxLeft + log2(leftovers[maxLen * j + queueStart]);
			} else {
				leftovers[maxLen * j + queueStart] = log2(0);
			}
		}
		buffer[maxLen * (context - 2) + queueStart] = 0;
		leftovers[maxLen * (context - 2) + queueStart] = log2(0);
		for (size_t j = 0; j < maxLen; j++) {
			if (scratch[j]) {
				buffer[maxLen * (context - 2) + queueStart] += (i + j + 1 == numCodepoints) ? (pow(2, -wrappedString[i].logTokenizations) * wrappedString[i].logTokenizations) : (pow(2, wrappedString[i + j + 1].logTokenizations - wrappedString[i].logTokenizations) * (-wrappedString[i + j + 1].logTokenizations + wrappedString[i].logTokenizations));
				if (i + j + 1 == numCodepoints) {
					leftovers[maxLen * (context - 2) + queueStart] = 0;
				}
			}
		}
		wrappedString[i].entropy /= context;
		for (size_t j = 0; j < context - 1; j++) {
			double subEntropy = pow(2, leftovers[maxLen * j + queueStart] - wrappedString[i].logTokenizations) * wrappedString[i].logTokenizations;
			wrappedString[i].entropy += subEntropy / (context - 1 - j) - subEntropy / context;
		}
		for (size_t j = 0; j < maxLen; j++) {
			scratch[j] = 0;
		}
		queueStart = (queueStart == 0) ? (maxLen - 1) : (queueStart - 1);
		i++;
	}
	free(leftovers);
	double sumExpVal = rand() / (double)((unsigned int)RAND_MAX + 1u);
	for (size_t i = 0; i < numCodepoints; i++) {
		if (wrappedString[i].valid) {
			for (size_t j = 0; j < (size_t)(wrappedString[i].entropy / log2(exp(1.0)) * scaling + 1); j++) {
				if (round(sumExpVal + pow(2, wrappedString[i].sampleLog + wrappedString[i].logTokenizations - wrappedString[0].logTokenizations)) - round(sumExpVal) != 0) {
					size_t vPos = i;
					for (size_t k = 0; k < context && vPos < numCodepoints - 1; k++) {
						double random = (double)rand() / ((unsigned int)RAND_MAX + 1u);
						size_t ind = 0;
						lastVocab = vocab;
						struct codepoint curPoint = wrappedString[vPos];
						char codeIndex = 0;
						size_t traveled = 0;
						size_t startTravel;
						while (vPos + traveled < numCodepoints - 1 && traveled < maxLen && lastVocab[(unsigned char)curPoint.content[codeIndex]] != NULL && random >= 0) {
							if (codeIndex == strlen(curPoint.content) - 1) {
								if (lastVocab[(unsigned char)curPoint.content[codeIndex]]->next[0] != NULL) {
									random -= pow(2, wrappedString[vPos + traveled + 1].logTokenizations - wrappedString[vPos].logTokenizations);
									ind = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next[0]->vocab;
								}
								lastVocab = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next;
								traveled++;
								curPoint = wrappedString[vPos + traveled];
								codeIndex = 0;
							} else {
								lastVocab = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next;
								codeIndex++;
							}
						}
						startTravel = traveled;
						if (random < 0) {
							if (!k && *startPoint > 0) {
								if (fwrite("\n", sizeof(char), 1, dump) != 1) {
									free(buffer);
									free(scratch);
									return -1;
								}
								(*startPoint)++;
							}
							if (k) {
								if (fwrite(",", sizeof(char), 1, dump) != 1) {
									free(buffer);
									free(scratch);
									return -1;
								}
								(*startPoint)++;
							}
							if (fwrite("\"", sizeof(char), 1, dump) != 1) {
								free(buffer);
								free(scratch);
								return -1;
							}
							(*startPoint)++;
							int numWritten;
							if ((numWritten = fprintf(dump, "%zu", ind)) < 0) {
								free(buffer);
								free(scratch);
								return -1;
							}
							*startPoint += numWritten;
							if (fwrite(",", sizeof(char), 1, dump) != 1) {
								free(buffer);
								free(scratch);
								return -1;
							}
							(*startPoint)++;
							size_t maxVocab = 0;
							lastVocab = vocab;
							curPoint = wrappedString[vPos + traveled];
							codeIndex = 0;
							while (vPos + traveled < numCodepoints && traveled < maxLen + startTravel && lastVocab[(unsigned char)curPoint.content[codeIndex]] != NULL) {
								if (codeIndex == strlen(curPoint.content) - 1) {
									if (lastVocab[(unsigned char)curPoint.content[codeIndex]]->next[0] != NULL) {
										if (vPos + traveled + 1 < numCodepoints) {
											if (wrappedString[vPos + startTravel].numTokens == wrappedString[vPos + traveled + 1].numTokens + 1) {
												maxVocab = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next[0]->vocab;
											}
										} else {
											if (wrappedString[vPos + startTravel].numTokens == 1) {
												maxVocab = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next[0]->vocab;
											}
										}
									}
									traveled++;
									if (vPos + traveled < numCodepoints) {
										lastVocab = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next;
										curPoint = wrappedString[vPos + traveled];
									}
									codeIndex = 0;
								} else {
									if (vPos + traveled < numCodepoints) {
										lastVocab = lastVocab[(unsigned char)curPoint.content[codeIndex]]->next;
									}
									codeIndex++;
								}
							}
							if ((numWritten = fprintf(dump, "%zu", maxVocab)) < 0) {
								free(buffer);
								free(scratch);
								return -1;
							}
							*startPoint += numWritten;
							if (fwrite("\"", sizeof(char), 1, dump) != 1) {
								free(buffer);
								free(scratch);
								return -1;
							}
							(*startPoint)++;
						}
						vPos += startTravel;
					}
				}
				sumExpVal += pow(2, wrappedString[i].sampleLog + wrappedString[i].logTokenizations - wrappedString[0].logTokenizations);
			}
		}
	}
	free(buffer);
	free(scratch);
	return 0;
}
size_t getVocab(FILE *vocabFile, struct node **vocabs, size_t *numVocabs) {
	for (int i = 0; i < 256; i++) {
		vocabs[i] = NULL;
	}
	struct node *curNode = NULL;
	char escaped = 1;
	char start = 0;
	char quoted = 0;
	char first = 0;
	long int dist = 0;
	long int maxDist = 0;
	long int vocLen = 0;
	char fileStart = 1;
	char bufDist = 0;
	char bufPoint = 0;
	size_t utfVal = 0;
	size_t vocCount = 0;
	while (!feof(vocabFile)) {
		int ch = getc(vocabFile);
		if (ch == EOF && !feof(vocabFile)) {
			struct node **curBufs = vocabs;
			struct node *backtrack = NULL;
			int i = 0;
			while (i < 256) {
				if (curBufs[i] != NULL) {
					backtrack = curBufs[i]->backtrack;
					curBufs = curBufs[i]->next;
					i = -1;
				}
				i++;
				if (i == 256 && curBufs != vocabs) {
					if (backtrack == NULL) {
						for (i = 0; vocabs[i]->next != curBufs; i++) {}
						free(vocabs[i]);
						vocabs[i] = NULL;
						curBufs = vocabs;
					} else {
						for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
						free(backtrack->next[i]);
						backtrack->next[i] = NULL;
						curBufs = backtrack->next;
						backtrack = backtrack->backtrack;
					}
				}
			}
			return -1;
		}
		if (fileStart && !feof(vocabFile) && ch == '"') {
			start = 1;
			fileStart = 0;
		} else if (fileStart && !feof(vocabFile)) {
			fileStart = 0;
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
			if (bufDist > 0 && bufPoint == bufDist && utfVal > 127 && utfVal < 1114112 && (((utfVal >> 7) && bufDist == 2) || ((utfVal >> 11) && bufDist == 3) || ((utfVal >> 16) && bufDist == 4))) {
				dist++;
			} else if (bufDist) {
				dist += bufPoint;
			}
			bufDist = 0;
			bufPoint = 0;
			utfVal = 0;
			if (dist > maxDist) {
				maxDist = dist;
			}
			curNode->next[0] = malloc(sizeof(struct node));
			if (curNode->next[0] == NULL) {
				struct node **curBufs = vocabs;
				struct node *backtrack = NULL;
				int i = 0;
				while (i < 256) {
					if (curBufs[i] != NULL) {
						backtrack = curBufs[i]->backtrack;
						curBufs = curBufs[i]->next;
						i = -1;
					}
					i++;
					if (i == 256 && curBufs != vocabs) {
						if (backtrack == NULL) {
							for (i = 0; vocabs[i]->next != curBufs; i++) {}
							free(vocabs[i]);
							vocabs[i] = NULL;
							curBufs = vocabs;
						} else {
							for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
							free(backtrack->next[i]);
							backtrack->next[i] = NULL;
							curBufs = backtrack->next;
							backtrack = backtrack->backtrack;
						}
					}
				}
				return -1;
			}
			curNode->next[0]->leaf = 0;
			curNode->next[0]->backtrack = curNode;
			for (int i = 0; i < 256; i++) {
				curNode->next[0]->next[i] = NULL;
			}
			curNode->next[0]->vocab = vocCount;
			struct node *nextNode = NULL;
			while (curNode != NULL) {
				int bCr = 0;
				if (curNode->backtrack != NULL) {
					for (; curNode->backtrack->next[bCr] != curNode; bCr++) {}
				} else {
					for (; vocabs[bCr] != curNode; bCr++) {}
				}
				if (nextNode == NULL) {
					if (vocabs[bCr] == NULL) {
						vocabs[bCr] = malloc(sizeof(struct node));
						if (vocabs[bCr] == NULL) {
							struct node **curBufs = vocabs;
							struct node *backtrack = NULL;
							int i = 0;
							while (i < 256) {
								if (curBufs[i] != NULL) {
									backtrack = curBufs[i]->backtrack;
									curBufs = curBufs[i]->next;
									i = -1;
								}
								i++;
								if (i == 256 && curBufs != vocabs) {
									if (backtrack == NULL) {
										for (i = 0; vocabs[i]->next != curBufs; i++) {}
										free(vocabs[i]);
										vocabs[i] = NULL;
										curBufs = vocabs;
									} else {
										for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
										free(backtrack->next[i]);
										backtrack->next[i] = NULL;
										curBufs = backtrack->next;
										backtrack = backtrack->backtrack;
									}
								}
							}
							return -1;
						}
						vocabs[bCr]->leaf = 0;
						vocabs[bCr]->backtrack = NULL;
						for (int i = 0; i < 256; i++) {
							vocabs[bCr]->next[i] = NULL;
						}
					}
					nextNode = vocabs[bCr];
				} else {
					if (nextNode->next[bCr] == NULL) {
						nextNode->next[bCr] = malloc(sizeof(struct node));
						if (nextNode->next[bCr] == NULL) {
							struct node **curBufs = vocabs;
							struct node *backtrack = NULL;
							int i = 0;
							while (i < 256) {
								if (curBufs[i] != NULL) {
									backtrack = curBufs[i]->backtrack;
									curBufs = curBufs[i]->next;
									i = -1;
								}
								i++;
								if (i == 256 && curBufs != vocabs) {
									if (backtrack == NULL) {
										for (i = 0; vocabs[i]->next != curBufs; i++) {}
										free(vocabs[i]);
										vocabs[i] = NULL;
										curBufs = vocabs;
									} else {
										for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
										free(backtrack->next[i]);
										backtrack->next[i] = NULL;
										curBufs = backtrack->next;
										backtrack = backtrack->backtrack;
									}
								}
							}
							return -1;
						}
						nextNode->next[bCr]->leaf = 0;
						nextNode->next[bCr]->backtrack = nextNode;
						for (int i = 0; i < 256; i++) {
							nextNode->next[bCr]->next[i] = NULL;
						}
					}
					nextNode = nextNode->next[bCr];
				}
				curNode = curNode->backtrack;
			}
			curNode = NULL;
			nextNode->leaf = 1;
			nextNode->vocab = vocCount;
			dist = 0;
			start = 1;
			quoted = 0;
			vocLen = 0;
			vocCount++;
		} else if (!feof(vocabFile)) {
			if (first) {
				first = 0;
			} else if (quoted) {
				if (escaped) {
					if (ch == '"') {
						if (bufDist > 0 && bufPoint < bufDist) {
							dist += bufPoint;
							bufDist = 0;
							bufPoint = 0;
							utfVal = 0;
						}
						dist++;
						escaped = 0;
						if (curNode == NULL) {
							if (vocabs[ch] == NULL) {
								vocabs[ch] = malloc(sizeof(struct node));
								if (vocabs[ch] == NULL) {
									struct node **curBufs = vocabs;
									struct node *backtrack = NULL;
									int i = 0;
									while (i < 256) {
										if (curBufs[i] != NULL) {
											backtrack = curBufs[i]->backtrack;
											curBufs = curBufs[i]->next;
											i = -1;
										}
										i++;
										if (i == 256 && curBufs != vocabs) {
											if (backtrack == NULL) {
												for (i = 0; vocabs[i]->next != curBufs; i++) {}
												free(vocabs[i]);
												vocabs[i] = NULL;
												curBufs = vocabs;
											} else {
												for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
												free(backtrack->next[i]);
												backtrack->next[i] = NULL;
												curBufs = backtrack->next;
												backtrack = backtrack->backtrack;
											}
										}
									}
									return -1;
								}
								vocabs[ch]->leaf = 0;
								vocabs[ch]->backtrack = NULL;
								for (int i = 0; i < 256; i++) {
									vocabs[ch]->next[i] = NULL;
								}
							}
							curNode = vocabs[ch];
						} else {
							if (curNode->next[ch] == NULL) {
								curNode->next[ch] = malloc(sizeof(struct node));
								if (curNode->next[ch] == NULL) {
									struct node **curBufs = vocabs;
									struct node *backtrack = NULL;
									int i = 0;
									while (i < 256) {
										if (curBufs[i] != NULL) {
											backtrack = curBufs[i]->backtrack;
											curBufs = curBufs[i]->next;
											i = -1;
										}
										i++;
										if (i == 256 && curBufs != vocabs) {
											if (backtrack == NULL) {
												for (i = 0; vocabs[i]->next != curBufs; i++) {}
												free(vocabs[i]);
												vocabs[i] = NULL;
												curBufs = vocabs;
											} else {
												for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
												free(backtrack->next[i]);
												backtrack->next[i] = NULL;
												curBufs = backtrack->next;
												backtrack = backtrack->backtrack;
											}
										}
									}
									return -1;
								}
								curNode->next[ch]->leaf = 0;
								curNode->next[ch]->backtrack = curNode;
								for (int i = 0; i < 256; i++) {
									curNode->next[ch]->next[i] = NULL;
								}
							}
							curNode = curNode->next[ch];
						}
						vocLen++;
					} else {
						struct node **curBufs = vocabs;
						struct node *backtrack = NULL;
						int i = 0;
						while (i < 256) {
							if (curBufs[i] != NULL) {
								backtrack = curBufs[i]->backtrack;
								curBufs = curBufs[i]->next;
								i = -1;
							}
							i++;
							if (i == 256 && curBufs != vocabs) {
								if (backtrack == NULL) {
									for (i = 0; vocabs[i]->next != curBufs; i++) {}
									free(vocabs[i]);
									vocabs[i] = NULL;
									curBufs = vocabs;
								} else {
									for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
									free(backtrack->next[i]);
									backtrack->next[i] = NULL;
									curBufs = backtrack->next;
									backtrack = backtrack->backtrack;
								}
							}
						}
						return -1;
					}
				} else {
					if (ch == '"') {
						escaped = 1;
					} else {
						if (bufPoint < bufDist && (ch & 192) == 128) {
							utfVal = (utfVal << 6) + (ch & 63);
							bufPoint++;
						} else {
							if (bufDist > 0 && bufPoint == bufDist && utfVal > 127 && utfVal < 1114112 && (((utfVal >> 7) && bufDist == 2) || ((utfVal >> 11) && bufDist == 3) || ((utfVal >> 16) && bufDist == 4))) {
								dist++;
							} else if (bufDist) {
								dist += bufPoint;
							}
							bufDist = 0;
							bufPoint = 0;
							utfVal = 0;
							if ((ch & 224) == 192) {
								bufDist = 2;
								bufPoint = 1;
								utfVal = ch & ((1 << (7 - bufDist)) - 1);
							} else if ((ch & 240) == 224) {
								bufDist = 3;
								bufPoint = 1;
								utfVal = ch & ((1 << (7 - bufDist)) - 1);
	 						} else if ((ch & 248) == 240) {
								bufDist = 4;
								bufPoint = 1;
								utfVal = ch & ((1 << (7 - bufDist)) - 1);
							} else {
								dist++;
							}
						}
						if (curNode == NULL) {
							if (vocabs[ch] == NULL) {
								vocabs[ch] = malloc(sizeof(struct node));
								if (vocabs[ch] == NULL) {
									struct node **curBufs = vocabs;
									struct node *backtrack = NULL;
									int i = 0;
									while (i < 256) {
										if (curBufs[i] != NULL) {
											backtrack = curBufs[i]->backtrack;
											curBufs = curBufs[i]->next;
											i = -1;
										}
										i++;
										if (i == 256 && curBufs != vocabs) {
											if (backtrack == NULL) {
												for (i = 0; vocabs[i]->next != curBufs; i++) {}
												free(vocabs[i]);
												vocabs[i] = NULL;
												curBufs = vocabs;
											} else {
												for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
												free(backtrack->next[i]);
												backtrack->next[i] = NULL;
												curBufs = backtrack->next;
												backtrack = backtrack->backtrack;
											}
										}
									}
									return -1;
								}
								vocabs[ch]->leaf = 0;
								vocabs[ch]->backtrack = NULL;
								for (int i = 0; i < 256; i++) {
									vocabs[ch]->next[i] = NULL;
								}
							}
							curNode = vocabs[ch];
						} else {
							if (curNode->next[ch] == NULL) {
								curNode->next[ch] = malloc(sizeof(struct node));
								if (curNode->next[ch] == NULL) {
									struct node **curBufs = vocabs;
									struct node *backtrack = NULL;
									int i = 0;
									while (i < 256) {
										if (curBufs[i] != NULL) {
											backtrack = curBufs[i]->backtrack;
											curBufs = curBufs[i]->next;
											i = -1;
										}
										i++;
										if (i == 256 && curBufs != vocabs) {
											if (backtrack == NULL) {
												for (i = 0; vocabs[i]->next != curBufs; i++) {}
												free(vocabs[i]);
												vocabs[i] = NULL;
												curBufs = vocabs;
											} else {
												for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
												free(backtrack->next[i]);
												backtrack->next[i] = NULL;
												curBufs = backtrack->next;
												backtrack = backtrack->backtrack;
											}
										}
									}
									return -1;
								}
								curNode->next[ch]->leaf = 0;
								curNode->next[ch]->backtrack = curNode;
								for (int i = 0; i < 256; i++) {
									curNode->next[ch]->next[i] = NULL;
								}
							}
							curNode = curNode->next[ch];
						}
						vocLen++;
					}
				}
			} else {
				if (ch == '"') {
					struct node **curBufs = vocabs;
					struct node *backtrack = NULL;
					int i = 0;
					while (i < 256) {
						if (curBufs[i] != NULL) {
							backtrack = curBufs[i]->backtrack;
							curBufs = curBufs[i]->next;
							i = -1;
						}
						i++;
						if (i == 256 && curBufs != vocabs) {
							if (backtrack == NULL) {
								for (i = 0; vocabs[i]->next != curBufs; i++) {}
								free(vocabs[i]);
								vocabs[i] = NULL;
								curBufs = vocabs;
							} else {
								for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
								free(backtrack->next[i]);
								backtrack->next[i] = NULL;
								curBufs = backtrack->next;
								backtrack = backtrack->backtrack;
							}
						}
					}
					return -1;
				}
				if (bufPoint < bufDist && (ch & 192) == 128) {
					utfVal = (utfVal << 6) + (ch & 63);
					bufPoint++;
				} else {
					if (bufDist > 0 && bufPoint == bufDist && utfVal > 127 && utfVal < 1114112 && (((utfVal >> 7) && bufDist == 2) || ((utfVal >> 11) && bufDist == 3) || ((utfVal >> 16) && bufDist == 4))) {
						dist++;
					} else if (bufDist) {
						dist += bufPoint;
					}
					bufDist = 0;
					bufPoint = 0;
					utfVal = 0;
					if ((ch & 224) == 192) {
						bufDist = 2;
						bufPoint = 1;
						utfVal = ch & ((1 << (7 - bufDist)) - 1);
					} else if ((ch & 240) == 224) {
						bufDist = 3;
						bufPoint = 1;
						utfVal = ch & ((1 << (7 - bufDist)) - 1);
	 				} else if ((ch & 248) == 240) {
						bufDist = 4;
						bufPoint = 1;
						utfVal = ch & ((1 << (7 - bufDist)) - 1);
					} else {
						dist++;
					}
				}
				if (curNode == NULL) {
					if (vocabs[ch] == NULL) {
						vocabs[ch] = malloc(sizeof(struct node));
						if (vocabs[ch] == NULL) {
							struct node **curBufs = vocabs;
							struct node *backtrack = NULL;
							int i = 0;
							while (i < 256) {
								if (curBufs[i] != NULL) {
									backtrack = curBufs[i]->backtrack;
									curBufs = curBufs[i]->next;
									i = -1;
								}
								i++;
								if (i == 256 && curBufs != vocabs) {
									if (backtrack == NULL) {
										for (i = 0; vocabs[i]->next != curBufs; i++) {}
										free(vocabs[i]);
										vocabs[i] = NULL;
										curBufs = vocabs;
									} else {
										for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
										free(backtrack->next[i]);
										backtrack->next[i] = NULL;
										curBufs = backtrack->next;
										backtrack = backtrack->backtrack;
									}
								}
							}
							return -1;
						}
						vocabs[ch]->leaf = 0;
						vocabs[ch]->backtrack = NULL;
						for (int i = 0; i < 256; i++) {
							vocabs[ch]->next[i] = NULL;
						}
					}
					curNode = vocabs[ch];
				} else {
					if (curNode->next[ch] == NULL) {
						curNode->next[ch] = malloc(sizeof(struct node));
						if (curNode->next[ch] == NULL) {
							struct node **curBufs = vocabs;
							struct node *backtrack = NULL;
							int i = 0;
							while (i < 256) {
								if (curBufs[i] != NULL) {
									backtrack = curBufs[i]->backtrack;
									curBufs = curBufs[i]->next;
									i = -1;
								}
								i++;
								if (i == 256 && curBufs != vocabs) {
									if (backtrack == NULL) {
										for (i = 0; vocabs[i]->next != curBufs; i++) {}
										free(vocabs[i]);
										vocabs[i] = NULL;
										curBufs = vocabs;
									} else {
										for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
										free(backtrack->next[i]);
										backtrack->next[i] = NULL;
										curBufs = backtrack->next;
										backtrack = backtrack->backtrack;
									}
								}
							}
							return -1;
						}
						curNode->next[ch]->leaf = 0;
						curNode->next[ch]->backtrack = curNode;
						for (int i = 0; i < 256; i++) {
							curNode->next[ch]->next[i] = NULL;
						}
					}
					curNode = curNode->next[ch];
				}
				vocLen++;
			}
		}
	}
	*numVocabs = vocCount;
	return maxDist;
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
	struct node **vocab = malloc(256 * sizeof(struct node*));
	if (vocab == NULL) {
		fclose(vocabFile);
		fclose(dump);
		return 1;
	}
	size_t numVocabs;
	size_t maxLen = getVocab(vocabFile, vocab, &numVocabs);
	if (maxLen == -1) {
		fclose(vocabFile);
		fclose(dump);
		free(vocab);
		return 1;
	}
	if (maxLen == 0) {
		maxLen = 1;
	}
	if (fclose(vocabFile) == EOF) {
		struct node **curBufs = vocab;
		struct node *backtrack = NULL;
		int i = 0;
		while (i < 256) {
			if (curBufs[i] != NULL) {
				backtrack = curBufs[i]->backtrack;
				curBufs = curBufs[i]->next;
				i = -1;
			}
			i++;
			if (i == 256 && curBufs != vocab) {
				if (backtrack == NULL) {
					for (i = 0; vocab[i]->next != curBufs; i++) {}
					free(vocab[i]);
					vocab[i] = NULL;
					curBufs = vocab;
				} else {
					for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
					free(backtrack->next[i]);
					backtrack->next[i] = NULL;
					curBufs = backtrack->next;
					backtrack = backtrack->backtrack;
				}
			}
		}
		free(vocab);
		fclose(dump);
		return 1;
	}
	vocabFile = fopen(argv[3], "wb+");
	if (vocabFile == NULL) {
		struct node **curBufs = vocab;
		struct node *backtrack = NULL;
		int i = 0;
		while (i < 256) {
			if (curBufs[i] != NULL) {
				backtrack = curBufs[i]->backtrack;
				curBufs = curBufs[i]->next;
				i = -1;
			}
			i++;
			if (i == 256 && curBufs != vocab) {
				if (backtrack == NULL) {
					for (i = 0; vocab[i]->next != curBufs; i++) {}
					free(vocab[i]);
					vocab[i] = NULL;
					curBufs = vocab;
				} else {
					for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
					free(backtrack->next[i]);
					backtrack->next[i] = NULL;
					curBufs = backtrack->next;
					backtrack = backtrack->backtrack;
				}
			}
		}
		free(vocab);
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
			struct node **curBufs = vocab;
			struct node *backtrack = NULL;
			int i = 0;
			while (i < 256) {
				if (curBufs[i] != NULL) {
					backtrack = curBufs[i]->backtrack;
					curBufs = curBufs[i]->next;
					i = -1;
				}
				i++;
				if (i == 256 && curBufs != vocab) {
					if (backtrack == NULL) {
						for (i = 0; vocab[i]->next != curBufs; i++) {}
						free(vocab[i]);
						vocab[i] = NULL;
						curBufs = vocab;
					} else {
						for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
						free(backtrack->next[i]);
						backtrack->next[i] = NULL;
						curBufs = backtrack->next;
						backtrack = backtrack->backtrack;
					}
				}
			}
			free(vocab);
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
						struct node **curBufs = vocab;
						struct node *backtrack = NULL;
						int i = 0;
						while (i < 256) {
							if (curBufs[i] != NULL) {
								backtrack = curBufs[i]->backtrack;
								curBufs = curBufs[i]->next;
								i = -1;
							}
							i++;
							if (i == 256 && curBufs != vocab) {
								if (backtrack == NULL) {
									for (i = 0; vocab[i]->next != curBufs; i++) {}
									free(vocab[i]);
									vocab[i] = NULL;
									curBufs = vocab;
								} else {
									for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
									free(backtrack->next[i]);
									backtrack->next[i] = NULL;
									curBufs = backtrack->next;
									backtrack = backtrack->backtrack;
								}
							}
						}
						free(vocab);
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
					struct node **curBufs = vocab;
					struct node *backtrack = NULL;
					int i = 0;
					while (i < 256) {
						if (curBufs[i] != NULL) {
							backtrack = curBufs[i]->backtrack;
							curBufs = curBufs[i]->next;
							i = -1;
						}
						i++;
						if (i == 256 && curBufs != vocab) {
							if (backtrack == NULL) {
								for (i = 0; vocab[i]->next != curBufs; i++) {}
								free(vocab[i]);
								vocab[i] = NULL;
								curBufs = vocab;
							} else {
								for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
								free(backtrack->next[i]);
								backtrack->next[i] = NULL;
								curBufs = backtrack->next;
								backtrack = backtrack->backtrack;
							}
						}
					}
					free(vocab);
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
				struct node **curBufs = vocab;
				struct node *backtrack = NULL;
				int i = 0;
				while (i < 256) {
					if (curBufs[i] != NULL) {
						backtrack = curBufs[i]->backtrack;
						curBufs = curBufs[i]->next;
						i = -1;
					}
					i++;
					if (i == 256 && curBufs != vocab) {
						if (backtrack == NULL) {
							for (i = 0; vocab[i]->next != curBufs; i++) {}
							free(vocab[i]);
							vocab[i] = NULL;
							curBufs = vocab;
						} else {
							for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
							free(backtrack->next[i]);
							backtrack->next[i] = NULL;
							curBufs = backtrack->next;
							backtrack = backtrack->backtrack;
						}
					}
				}
				free(vocab);
				fclose(vocabFile);
				fclose(dump);
				return 1;
			}
			if (fseek(dump, -1 * numCodepoints * sizeof(struct codepoint), SEEK_CUR) < 0) {
				free(wrappedString);
				struct node **curBufs = vocab;
				struct node *backtrack = NULL;
				int i = 0;
				while (i < 256) {
					if (curBufs[i] != NULL) {
						backtrack = curBufs[i]->backtrack;
						curBufs = curBufs[i]->next;
						i = -1;
					}
					i++;
					if (i == 256 && curBufs != vocab) {
						if (backtrack == NULL) {
							for (i = 0; vocab[i]->next != curBufs; i++) {}
							free(vocab[i]);
							vocab[i] = NULL;
							curBufs = vocab;
						} else {
							for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
							free(backtrack->next[i]);
							backtrack->next[i] = NULL;
							curBufs = backtrack->next;
							backtrack = backtrack->backtrack;
						}
					}
				}
				free(vocab);
				fclose(vocabFile);
				fclose(dump);
				return 1;
			}
			if (fread(wrappedString, sizeof(struct codepoint), numCodepoints, dump) != numCodepoints) {
				free(wrappedString);
				struct node **curBufs = vocab;
				struct node *backtrack = NULL;
				int i = 0;
				while (i < 256) {
					if (curBufs[i] != NULL) {
						backtrack = curBufs[i]->backtrack;
						curBufs = curBufs[i]->next;
						i = -1;
					}
					i++;
					if (i == 256 && curBufs != vocab) {
						if (backtrack == NULL) {
							for (i = 0; vocab[i]->next != curBufs; i++) {}
							free(vocab[i]);
							vocab[i] = NULL;
							curBufs = vocab;
						} else {
							for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
							free(backtrack->next[i]);
							backtrack->next[i] = NULL;
							curBufs = backtrack->next;
							backtrack = backtrack->backtrack;
						}
					}
				}
				free(vocab);
				fclose(dump);
				fclose(vocabFile);
				return 1;
			}
			if (fseek(dump, -1 * numCodepoints * sizeof(struct codepoint), SEEK_CUR) < 0) {
				free(wrappedString);
				struct node **curBufs = vocab;
				struct node *backtrack = NULL;
				int i = 0;
				while (i < 256) {
					if (curBufs[i] != NULL) {
						backtrack = curBufs[i]->backtrack;
						curBufs = curBufs[i]->next;
						i = -1;
					}
					i++;
					if (i == 256 && curBufs != vocab) {
						if (backtrack == NULL) {
							for (i = 0; vocab[i]->next != curBufs; i++) {}
							free(vocab[i]);
							vocab[i] = NULL;
							curBufs = vocab;
						} else {
							for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
							free(backtrack->next[i]);
							backtrack->next[i] = NULL;
							curBufs = backtrack->next;
							backtrack = backtrack->backtrack;
						}
					}
				}
				free(vocab);
				fclose(vocabFile);
				fclose(dump);
				return 1;
			}
			if (generateSamplesFromString(dump, vocabFile, &numVocabs, maxLen, wrappedString, numCodepoints, vocab, atoi(argv[4]), strtod(argv[5], NULL), &startPoint) == -1) {
				free(wrappedString);
				struct node **curBufs = vocab;
				struct node *backtrack = NULL;
				int i = 0;
				while (i < 256) {
					if (curBufs[i] != NULL) {
						backtrack = curBufs[i]->backtrack;
						curBufs = curBufs[i]->next;
						i = -1;
					}
					i++;
					if (i == 256 && curBufs != vocab) {
						if (backtrack == NULL) {
							for (i = 0; vocab[i]->next != curBufs; i++) {}
							free(vocab[i]);
							vocab[i] = NULL;
							curBufs = vocab;
						} else {
							for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
							free(backtrack->next[i]);
							backtrack->next[i] = NULL;
							curBufs = backtrack->next;
							backtrack = backtrack->backtrack;
						}
					}
				}
				free(vocab);
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
						struct node **curBufs = vocab;
						struct node *backtrack = NULL;
						int i = 0;
						while (i < 256) {
							if (curBufs[i] != NULL) {
								backtrack = curBufs[i]->backtrack;
								curBufs = curBufs[i]->next;
								i = -1;
							}
							i++;
							if (i == 256 && curBufs != vocab) {
								if (backtrack == NULL) {
									for (i = 0; vocab[i]->next != curBufs; i++) {}
									free(vocab[i]);
									vocab[i] = NULL;
									curBufs = vocab;
								} else {
									for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
									free(backtrack->next[i]);
									backtrack->next[i] = NULL;
									curBufs = backtrack->next;
									backtrack = backtrack->backtrack;
								}
							}
						}
						free(vocab);
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
							struct node **curBufs = vocab;
							struct node *backtrack = NULL;
							int i = 0;
							while (i < 256) {
								if (curBufs[i] != NULL) {
									backtrack = curBufs[i]->backtrack;
									curBufs = curBufs[i]->next;
									i = -1;
								}
								i++;
								if (i == 256 && curBufs != vocab) {
									if (backtrack == NULL) {
										for (i = 0; vocab[i]->next != curBufs; i++) {}
										free(vocab[i]);
										vocab[i] = NULL;
										curBufs = vocab;
									} else {
										for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
										free(backtrack->next[i]);
										backtrack->next[i] = NULL;
										curBufs = backtrack->next;
										backtrack = backtrack->backtrack;
									}
								}
							}
							free(vocab);
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
						struct node **curBufs = vocab;
						struct node *backtrack = NULL;
						int i = 0;
						while (i < 256) {
							if (curBufs[i] != NULL) {
								backtrack = curBufs[i]->backtrack;
								curBufs = curBufs[i]->next;
								i = -1;
							}
							i++;
							if (i == 256 && curBufs != vocab) {
								if (backtrack == NULL) {
									for (i = 0; vocab[i]->next != curBufs; i++) {}
									free(vocab[i]);
									vocab[i] = NULL;
									curBufs = vocab;
								} else {
									for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
									free(backtrack->next[i]);
									backtrack->next[i] = NULL;
									curBufs = backtrack->next;
									backtrack = backtrack->backtrack;
								}
							}
						}
						free(vocab);
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
		struct node **curBufs = vocab;
		struct node *backtrack = NULL;
		int i = 0;
		while (i < 256) {
			if (curBufs[i] != NULL) {
				backtrack = curBufs[i]->backtrack;
				curBufs = curBufs[i]->next;
				i = -1;
			}
			i++;
			if (i == 256 && curBufs != vocab) {
				if (backtrack == NULL) {
					for (i = 0; vocab[i]->next != curBufs; i++) {}
					free(vocab[i]);
					vocab[i] = NULL;
					curBufs = vocab;
				} else {
					for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
					free(backtrack->next[i]);
					backtrack->next[i] = NULL;
					curBufs = backtrack->next;
					backtrack = backtrack->backtrack;
				}
			}
		}
		free(vocab);
		fclose(vocabFile);
		fclose(dump);
		return 1;
	}
	struct node **curBufs = vocab;
	struct node *backtrack = NULL;
	int i = 0;
	while (i < 256) {
		if (curBufs[i] != NULL) {
			backtrack = curBufs[i]->backtrack;
			curBufs = curBufs[i]->next;
			i = -1;
		}
		i++;
		if (i == 256 && curBufs != vocab) {
			if (backtrack == NULL) {
				for (i = 0; vocab[i]->next != curBufs; i++) {}
				free(vocab[i]);
				vocab[i] = NULL;
				curBufs = vocab;
			} else {
				for (i = 0; backtrack->next[i]->next != curBufs; i++) {}
				free(backtrack->next[i]);
				backtrack->next[i] = NULL;
				curBufs = backtrack->next;
				backtrack = backtrack->backtrack;
			}
		}
	}
	free(vocab);
	if (fclose(vocabFile) == EOF) {
		fclose(dump);
		return 1;
	}
	if (fclose(dump) == EOF) {
		return 1;
	}
	return 0;
}