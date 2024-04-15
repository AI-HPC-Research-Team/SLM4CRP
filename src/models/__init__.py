from models.chemT5 import *

from transformers import T5Tokenizer, AutoTokenizer

SUPPORTED_Model = {
    "chemt5": ChemT5
}

SUPPORTED_Tokenizer = {
    "chemt5": T5Tokenizer
}

ckpt_folder = ""

SUPPORTED_CKPT = {
    "chemt5": ckpt_folder+"text_ckpts/ChemT5-base-augm"
}

class PromptManager:
    def __init__(self, config = None):
        self.config = config
        type_prompts = {
            'reagent': ['Based on the given chemical reaction, can you propose some likely reagents that might have been utilized? The reaction type is ', 
                            'Can you provide potential reagents for the following chemical reaction? The reaction type is ', 
                            'Please suggest some possible reagents that could have been used in the following chemical reaction. Given that the reaction type is ', 
                            'Given this chemical reaction, what are some reagents that could have been used? The reaction type is ', 
                            'Can you suggest some reagents that might have been used in the given chemical reaction? The reaction type is ', 
                            'Based on the given chemical reaction, suggest some possible reagents. What would be the outcome if the reaction type is ', 
                            'Given the following chemical reaction, what are some potential reagents that could have been employed? The reaction type is ', 
                            'Please propose potential reagents that might have been utilized in the provided chemical reaction. Given that the reaction type is ', 
                            'From the provided chemical reaction, propose some possible reagents that could have been used. Considering the reaction type is ', 
                            'Please provide possible reagents based on the following chemical reaction. What would be the outcome if the reaction type is ', 
                            'What reagents could have been utilized in the following chemical reaction? The reaction type is ', 
                            'Given the following reaction, what are some possible reagents that could have been utilized? Considering the reaction type is '],
            'retro':  ['Provided the product below, propose some possible reactants that could have been used in the reaction. Considering the reaction type is ', 
                           'Please suggest potential reactants used in the synthesis of the provided product. What would be the outcome if the reaction type is ', 
                           'Given these product, can you propose the corresponding reactants? Considering the reaction type is ', 
                           'What reactants could lead to the production of the following product? Considering the reaction type is ', 
                           'What are the possible reactants that could have formed the following product? Considering the reaction type is ', 
                           'Which reactants could have been used to generate the given product? Given that the reaction type is ', 
                           'Given the following product, please provide possible reactants. What would be the outcome if the reaction type is ', 
                           'Based on the given product, provide some plausible reactants that might have been utilized to prepare it. Considering the reaction type is ', 
                           'Can you identify some reactants that might result in the given product? Considering the reaction type is ', 
                           'With the provided product, recommend some probable reactants that were likely used in its production. The reaction type is ', 
                           'With the given product, suggest some likely reactants that were used in its synthesis. What would be the outcome if the reaction type is '],
            'forward':  ['Please suggest a potential product based on the given reactants and reagents. What would be the outcome if the reaction type is ', 
                             'Please provide a feasible product that could be formed using the given reactants and reagents. What would be the outcome if the reaction type is ', 
                             'Based on the given reactants and reagents, what product could potentially be produced? Considering the reaction type is ', 
                             'Given the reactants and reagents provided, what is a possible product that can be formed? Considering the reaction type is ', 
                             'Using the provided reactants and reagents, can you propose a likely product? Given that the reaction type is ', 
                             'Based on the given reactants and reagents, suggest a possible product. Given that the reaction type is ', 
                             'With the provided reactants and reagents, propose a potential product. Given that the reaction type is ', 
                             'Given the reactants and reagents below, come up with a possible product. Considering the reaction type is ', 
                             'Given the following reactants and reagents, please provide a possible product. What would be the outcome if the reaction type is ', 
                             'Using the listed reactants and reagents, offer a plausible product. What would be the outcome if the reaction type is ', 
                             'Given the reactants and reagents listed, what could be a probable product of their reaction? Given that the reaction type is ', 
                             'What product could potentially form from the reaction of the given reactants and reagents? The reaction type is ']
        }

   
        prompts = {
            'reagent': ['Based on the given chemical reaction, can you propose some likely reagents that might have been utilized?', 
                            'Can you provide potential reagents for the following chemical reaction?', 
                            'Please suggest some possible reagents that could have been used in the following chemical reaction.', 
                            'Given this chemical reaction, what are some reagents that could have been used?', 
                            'Can you suggest some reagents that might have been used in the given chemical reaction?', 
                            'Based on the given chemical reaction, suggest some possible reagents.', 
                            'Given the following chemical reaction, what are some potential reagents that could have been employed?', 
                            'Please propose potential reagents that might have been utilized in the provided chemical reaction.', 
                            'From the provided chemical reaction, propose some possible reagents that could have been used.', 
                            'Please provide possible reagents based on the following chemical reaction.', 
                            'What reagents could have been utilized in the following chemical reaction?', 
                            'Given the following reaction, what are some possible reagents that could have been utilized?'],
            'retro':  ['Provided the product below, propose some possible reactants that could have been used in the reaction.', 
                           'Please suggest potential reactants used in the synthesis of the provided product.', 
                           'Given these product, can you propose the corresponding reactants?', 
                           'What reactants could lead to the production of the following product?', 
                           'What are the possible reactants that could have formed the following product?', 
                           'Which reactants could have been used to generate the given product?', 
                           'Given the following product, please provide possible reactants.', 
                           'Based on the given product, provide some plausible reactants that might have been utilized to prepare it.', 
                           'Can you identify some reactants that might result in the given product?', 
                           'With the provided product, recommend some probable reactants that were likely used in its production.', 
                           'With the given product, suggest some likely reactants that were used in its synthesis.'],
            'forward':  ['Please suggest a potential product based on the given reactants and reagents.', 
                             'Please provide a feasible product that could be formed using the given reactants and reagents.', 
                             'Based on the given reactants and reagents, what product could potentially be produced?', 
                             'Given the reactants and reagents provided, what is a possible product that can be formed?', 
                             'Using the provided reactants and reagents, can you propose a likely product?', 
                             'Based on the given reactants and reagents, suggest a possible product.', 
                             'With the provided reactants and reagents, propose a potential product.', 
                             'Given the reactants and reagents below, come up with a possible product.', 
                             'Given the following reactants and reagents, please provide a possible product.', 
                             'Using the listed reactants and reagents, offer a plausible product.', 
                             'Given the reactants and reagents listed, what could be a probable product of their reaction?', 
                             'What product could potentially form from the reaction of the given reactants and reagents?']
        }

        if(config == None):
            self.prompts = type_prompts
        elif(config.reaction_type == True):
            self.prompts = type_prompts
        else:
            self.prompts = prompts
            
        self.prompts_list = self.get_all_prompts_by_type(self.config.task)
        
    def get_prompt(self, task_type, number):

        if task_type not in self.prompts:
            return "Invalid task type. Please choose from 'reagent', 'retro', or 'forward'."
        
  
        if number < 1 or number > len(self.prompts[task_type]):
            return "Invalid number. Please choose a number within the valid range for the chosen task type."
        

        return self.prompts[task_type][number-1]  

    def get_all_prompts_by_type(self, task_type):

        if task_type == 'reactions' :
      
            sorted_prompts = []
            
            order = ['forward', 'reagent', 'retro']
            for key in order:
                sorted_prompts.extend(self.prompts[key])
            return sorted_prompts
        elif task_type not in self.prompts:
            return "Invalid task type. Please choose from 'Reagent Prediction', 'Retrosynthesis', or 'Forward Reaction Prediction'."
        else:
       
            return self.prompts[task_type]
    
    def get_prompts_by_indices(self, indices):

        task_prompts = self.prompts_list

       
        selected_prompts = [task_prompts[i] for i in indices if i < len(task_prompts)]
        
        return selected_prompts