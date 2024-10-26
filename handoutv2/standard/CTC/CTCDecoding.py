import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        B = y_probs.shape[2]
        
        for t in range(len(y_probs[0])):
            max_r = -1
            max_prob = 0
            for r in range(len(y_probs)):
                if y_probs[r, t, 0] > max_prob:
                    max_prob = y_probs[r,t,0]
                    max_r = r
            if max_r == blank:
                decoded_path.append("BLANK")
            else:
                decoded_path.append(self.symbol_set[max_r-1])
            path_prob *= max_prob
            
        #Compress 
        ptr = len(decoded_path)-1
        while ptr > 0:
            if decoded_path[ptr] == decoded_path[ptr-1]:
                del decoded_path[ptr]
            ptr -= 1
        decoded_nonblank = list(filter(("BLANK").__ne__, decoded_path))
        result = ''.join(decoded_nonblank)
        
        return result, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        B = y_probs.shape[2]
        S = y_probs.shape[0]-1 #WITHOUT BLANK
        
        def initPaths(symbolSet, yProbs_t0):
            '''
            Generates initial paths (BATCHED)
            See L16 P164
            '''
            #symbolset: self.symbolset (all symbols without blank)
            # Shape: list(str)
            #yProbs_t0: y probabilities at t0
            # Shape: (len(symbols)+1, batch_size)
            assert yProbs_t0.shape == (S + 1, B)
            #Return: 
            # newPathsTerminalBlank, 
            # newPathsTerminalSym, 
            # newBlankPathScore, 
            # newPathScore
            initBlankPathScore = [dict() for _ in range(B)]
            initPathsWithFinalBlank = [list() for _ in range(B)]
            initPathScore = [dict() for _ in range(B)]
            path = ""
            # Assign scores of blanks
            for b in range(B):
                initBlankPathScore[b][path] = (yProbs_t0[0,b])
                initPathsWithFinalBlank[b].append(path)
            
            #Path ending with symbol.
            initPathsWithFinalSym = [list() for _ in range(B)]
            for b in range(B):
                for s in range(S):
                    sym = symbolSet[s]
                    initPathScore[b][sym] = yProbs_t0[s+1, b] #since 0th is blank
                    initPathsWithFinalSym[b].append(sym)
            assert len(initPathsWithFinalBlank) == B
            assert len(initPathsWithFinalSym) == B
            print("InitPaths: FB Path " + str(initPathsWithFinalBlank) + ", FS Path " + str(initPathsWithFinalSym) + ", initBScore " + str(initBlankPathScore) + ", initPScore " + str(initPathScore))
            return initPathsWithFinalBlank, initPathsWithFinalSym, initBlankPathScore, initPathScore
        
        def prune(pathTerminalBlank, pathTerminalSym, blankPathScore, pathScore, beamWidth):
            '''
            Prunes beam search tree based on beamWidth
            See L16 P167
            '''
            # print(blankPathScore)
            scoreList = [list() for _ in range(B)]
            prunedPathsTerminalBlank = [list() for _ in range(B)]
            prunedPathsTerminalSym = [list() for _ in range(B)]
            prunedBlankPathScores = [dict() for _ in range(B)]
            prunedPathScores = [dict() for _ in range(B)]
            print("Enter prune: TB Path " + str(pathTerminalBlank) + ", TS Path " + str(pathTerminalSym) + ", BScore " + str(blankPathScore) + ", SScore " + str(pathScore))
            print("-beamwidth " + str(beamWidth))
            for b in range(B):
                for x in blankPathScore[b]:
                    scoreList[b].append(blankPathScore[b][x]) 
                for x in pathScore[b]:
                    scoreList[b].append(pathScore[b][x])
                scoreList[b].sort()
                # find cutoff
                cutoff_b = scoreList[b][0] if (beamWidth > len(scoreList[b])) else scoreList[b][len(scoreList[b])-beamWidth]
                
                print("Cutoff: " + str(cutoff_b))
                for i in pathTerminalBlank[b]:
                    if blankPathScore[b][i] >= cutoff_b:
                        prunedPathsTerminalBlank[b].append(i)
                        prunedBlankPathScores[b][i] = (blankPathScore[b][i])
                for i in pathTerminalSym[b]:
                    if pathScore[b][i] >= cutoff_b:
                        prunedPathsTerminalSym[b].append(i)
                        prunedPathScores[b][i] = (pathScore[b][i])
            print("Exit prune: TB Path " + str(prunedPathsTerminalBlank) + ", TS Path " + str(prunedPathsTerminalSym) + ", BScore " + str(prunedBlankPathScores) + ", SScore " + str(prunedPathScores))
            return prunedPathsTerminalBlank, prunedPathsTerminalSym, prunedBlankPathScores, prunedPathScores
        
        def extendBlank(pathTerminalBlank, pathTerminalSym, blankPathScore, yprobs_t):
            updatedPathTerminalBlank = [list() for _ in range(B)]
            updatedBlankPathScore = [dict() for _ in range(B)]
            print("ExtendBlank: TB Path " + str(pathTerminalBlank) + ", TS Path " + str(pathTerminalSym))
            for b in range(B):
                for path in pathTerminalBlank[b]:
                    updatedPathTerminalBlank[b].append(path)
                    print("Extending path [" + str(path) + "]")
                    updatedBlankPathScore[b][path] = blankPathScore[b][path] * yprobs_t[0,b]
                    print("Before: " + str(blankPathScore[b][path]) + ", mul: " + str(yprobs_t[0,b]) + ", after: " + str(updatedBlankPathScore[b][path]))
                for path in pathTerminalSym[b]:
                    
                    if path in updatedPathTerminalBlank[b]:
                        updatedBlankPathScore[b][path] += pathScore[b][path] * yprobs_t[0,b]
                    else:
                        # print("--Path " + str(path) + " is NOT in " + str(updatedPathTerminalBlank[b]))
                        updatedPathTerminalBlank[b].append(path)
                        updatedBlankPathScore[b][path] = pathScore[b][path] * yprobs_t[0,b]
            print("Exit extendBlank: PathTerminalBlank " + str(updatedPathTerminalBlank) + ", blankPathScore " + str(updatedBlankPathScore))
            return updatedPathTerminalBlank, updatedBlankPathScore
            
        def extendSym(pathTerminalBlank, pathTerminalSym, blankPathScore, pathScore, symbolSet, yprobs_t):
            updatedPathsTerminalSymbol = [list() for _ in range(B)]
            updatedPathScore = [dict() for _ in range(B)]
            print("ExtendSym: TB Path " + str(pathTerminalBlank) + ", TS Path " + str(pathTerminalSym))
            for b in range(B):
                for path in pathTerminalBlank[b]:
                    for s_i in range(len(symbolSet)):
                        s = symbolSet[s_i]
                        newpath = path + s
                        updatedPathsTerminalSymbol[b].append(newpath)
                        updatedPathScore[b][newpath] = blankPathScore[b][path] * yprobs_t[s_i+1, b]
                print(yprobs_t)
                for path in pathTerminalSym[b]:
                    for s_i in range(len(symbolSet)):
                        s = symbolSet[s_i]
                        newpath = path if (s == path[-1]) else path+s
                        if newpath in updatedPathsTerminalSymbol[b]:
                            updatedPathScore[b][newpath] += pathScore[b][path] * yprobs_t[s_i+1, b]
                        else:
                            print("--Path " + str(newpath) + " is NOT in " + str(updatedPathsTerminalSymbol[b]))
                            updatedPathsTerminalSymbol[b].append(newpath)
                            updatedPathScore[b][newpath] = pathScore[b][path] * yprobs_t[s_i+1, b]
            print("Exit extendSym: pathTerminalSym " + str(updatedPathsTerminalSymbol) + ", pathScore " + str(updatedPathScore))
            return updatedPathsTerminalSymbol, updatedPathScore
        
        def mergeIdenticalPaths(pathTerminalBlank, pathTerminalSym, blankPathScore, pathScore):
            mergedPaths = pathTerminalSym
            finalPathScore = pathScore
            for b in range(B):
                for p in pathTerminalBlank[b]:
                    if p in mergedPaths[b]:
                        finalPathScore[b][p] += blankPathScore[b][p]
                    else:
                        mergedPaths[b].append(p)
                        finalPathScore[b][p] = blankPathScore[b][p]
            print("Exit mergeIDPaths: mergedPaths " + str(mergedPaths) + ", finalPathScore " + str(finalPathScore))
            return mergedPaths, finalPathScore
            
        def argmax_paths(mergedPaths, mergedPathScores):
            bestPath = []
            finalPathScore = []
            for b in range(B):
                path_best = None
                score_best = 0
                for path in mergedPaths[b]:
                    if mergedPathScores[b][path] > score_best:
                        score_best = mergedPathScores[b][path]
                        path_best = path
                bestPath.append(path_best)
                finalPathScore.append(score_best)
            print("Exit argmax: bestPath " + str(bestPath) + ", finalPathScore " + str(finalPathScore))
            return bestPath, finalPathScore
            
        newPathsTerminalBlank, newPathsTerminalSym, newBlankPathScore, newPathScore\
            = initPaths(self.symbol_set, y_probs[:,0,:])
        
        for t in range(1,T):
            # Prune to BeamWidth
            print("Time " + str(t))
            pathsWithTerminalBlank, pathsWithTerminalSym, blankPathScore, pathScore\
                = prune(newPathsTerminalBlank, newPathsTerminalSym, newBlankPathScore, newPathScore, self.beam_width)
            
            newPathsTerminalBlank, newBlankPathScore = extendBlank(pathsWithTerminalBlank, pathsWithTerminalSym, blankPathScore, y_probs[:,t,:])
            newPathsTerminalSym, newPathScore = extendSym(pathsWithTerminalBlank, pathsWithTerminalSym, blankPathScore, pathScore, self.symbol_set, y_probs[:,t,:])
        
        mergedPaths, mergedPathScores = mergeIdenticalPaths(newPathsTerminalBlank, newPathsTerminalSym, newBlankPathScore, newPathScore)
        
        bestPath, finalPathScore = argmax_paths(mergedPaths, mergedPathScores)
        return bestPath[0], mergedPathScores[0]