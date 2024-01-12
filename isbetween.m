function BOOL = isbetween(signalT,StartT,EndT)   
% finds if a data time point lies within the given combinations of StartTs and EndTs
% Inputs:
% signalT: array of times [1xT] or [Tx1]
% startT: array of times [1xT] or [Tx1]
% EndT: array of times [1xT] or [Tx1]
% Outputs:
% BOOL: 1 if signlT is between given combinations of StartTs and EndTs, 0
% otherwis

    BOOL=0;
    for i=1:length(StartT)
        if signalT>StartT(i) && signalT<EndT(i)
            BOOL=1;
        end
    end
end
