function T = taper_tf(X, A, vers)
   
    if(vers==1)
        T = sqrt(abs(X)) ./ sqrt(abs(A));
    elseif(vers==2)
        T = abs(X) ./ abs(A);
    elseif(vers==3)
        T = abs(x).^(1/3) ./ abs(A).^(1/3);
    elseif(vers==4)
        T = abs(x).^(1/4) ./ abs(A).^(1/4);
    end
 
    T = min(max(T,0),1);
end