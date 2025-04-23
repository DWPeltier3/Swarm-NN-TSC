function TranslatedChildren = collectUnusedChildrenObjects(ObjectLoaderResult)
    %disp('starting')
    TranslatedChildren = containers.Map;
    collectUnusedChildrenObjectsHelper(ObjectLoaderResult, TranslatedChildren, '');
    end
    
    function collectUnusedChildrenObjectsHelper(ObjectLoaderResult, ...
                                                newMap, ...
                                                prefix)
    for name = ObjectLoaderResult.Children.keys
        if isempty(prefix)  
            fullName = name{1}; 
        else 
            fullName = [prefix '/' name{1}];
        end
        if ObjectLoaderResult.Children(name{1}).HasFcn
            newMap(name{1}) = ObjectLoaderResult.Children(name{1}); 
            
        else
            collectUnusedChildrenObjectsHelper(ObjectLoaderResult.Children(name{1}), newMap, fullName); 
        end 
    end

end 
