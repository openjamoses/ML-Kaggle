""" Local implementations of grid search and randomized search """
from random import randint, seed
import time
seed( 42 )

def lineSearch(model, Xtrain, Ytrain, Xvalid, Yvalid, params, score, score_kwargs = {}):
    """
    Line search over a single Hyperparameter space.
    """
    
    valid_scores = []
    studied_params = []
    
    if len(params) == 1:
        param = list(params.keys())[0]
        for value in params[ param ]:
            
            model.set_params(**{param : value})
            model.fit(Xtrain, Ytrain)
            valid_scores.append( score(Yvalid, model.predict(Xvalid), 
                                                        **score_kwargs) )
            studied_params.append( {param : value} )
            print(f"Score : {100 * valid_scores[-1]:.4f} %", studied_params[-1])
            print("")
            
    else:
        raise ValueError(" Must have a single hyperparameter")
    
    return valid_scores, studied_params



def randomSearch(model, Xtrain, Ytrain, Xvalid, Yvalid, params, score, 
                                                        score_kwargs = {}, n_choices = 100):
    """
    Random search over Hyperparameter space.
    """
    
    valid_scores = []
    studied_params = []
    params_names = list( params.keys() )

    
    for i in range( n_choices ):
        dictio = {}
        for param in params_names:
            dictio[ param ] = \
                            params[param][ randint(0, len(params[param]) - 1) ]
            
        model.set_params( **dictio )
        start = time.time()
        model.fit(Xtrain, Ytrain)
        valid_scores.append( score(Yvalid, model.predict(Xvalid),
                                                         **score_kwargs) )
        studied_params.append( dictio )
        print(f"Score : {100 * valid_scores[-1]:.4f} %", studied_params[-1])
        print(f"Done in {(time.time() - start) / 60:.2f} minutes")
        print("")
    
    return valid_scores, studied_params
            
            
            
def save_submission(root_path, Ypred):
    """
    Save preductions in csv file for submission
    """
    with open(root_path + "submission.csv", 'w') as file:
        file.write("Id,Category\n")
        for i in range(Ypred.shape[0]):
            file.write(f"{i},{Ypred[i]}\n")
