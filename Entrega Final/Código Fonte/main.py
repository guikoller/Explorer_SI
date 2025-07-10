# --- Bibliotecas ---
import sys
import os
import time

# --- Módulos Internos ---
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer

# --- Função Principal ---
def main(data_folder_name, config_ag_folder_name):
    """
    Ponto de entrada principal da simulação.

    Configura o ambiente, instancia os agentes (Rescuer mestre e Explorers)
    e inicia o loop de simulação.
    """
    
    # --- Configuração de Caminhos ---
    # Define os caminhos absolutos para as pastas de configuração e de dados do ambiente.
    current_folder = os.path.abspath(os.getcwd())
    config_ag_folder = os.path.abspath(os.path.join(current_folder, config_ag_folder_name))
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))
    
    # --- Instanciação do Ambiente ---
    env = Env(data_folder)
    
    # --- Instanciação dos Agentes ---
    
    # 1. Instancia o Rescuer Mestre
    # Este agente é responsável por centralizar as informações dos exploradores e
    # coordenar a equipe de resgate. Ele precisa ser criado antes dos exploradores
    # para que eles possam ter uma referência a ele.
    rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
    # O 4 indica que ele deve esperar por 4 agentes exploradores.
    master_rescuer = Rescuer(env, rescuer_file, 4)

    # 2. Instancia os Agentes Exploradores
    # Vetores de prioridade de movimento para cada explorador.
    # Isso diversifica a estratégia de exploração, fazendo com que cada agente
    # prefira direções diferentes, reduzindo a sobreposição.
    priorities_vector = {
        1: [2, 1, 0, 7, 6, 5, 4, 3],  # Prioridade: Direita, Cima-Direita, Cima, ...
        2: [2, 3, 4, 5, 6, 7, 0, 1],  # Prioridade: Direita, Baixo-Direita, Baixo, ...
        3: [6, 7, 0, 1, 2, 3, 4, 5],  # Prioridade: Esquerda, Cima-Esquerda, Cima, ...
        4: [6, 5, 4, 3, 2, 1, 0, 7]   # Prioridade: Esquerda, Baixo-Esquerda, Baixo, ...
    }

    # Cria cada um dos 4 agentes exploradores
    for i in range(1, 5):
        explorer_file = os.path.join(config_ag_folder, f"explorer_{i}_config.txt")
        # Cada explorador recebe uma referência ao rescuer mestre e seu vetor de prioridade.
        Explorer(env, explorer_file, master_rescuer, priorities_vector[i])

    # --- Início da Simulação ---
    # Roda o loop principal do ambiente, que gerencia o ciclo de vida dos agentes.
    print("Simulação iniciada. Agentes estão em execução...")
    env.run()
    print("Simulação terminada.")
        
# --- Ponto de Execução do Script ---
if __name__ == '__main__':
    """
    Este bloco é executado quando o script é chamado diretamente.
    Ele permite configurar as pastas de dados e de configuração via linha de comando.
    """
    # Define as pastas padrão se nenhum argumento for passado.
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
        config_ag_folder_name = sys.argv[2] if len(sys.argv) > 2 else os.path.join("", "cfg")
    else:
        # Caminhos padrão para os dados e configurações
        data_folder_name = os.path.join("datasets", "data_408v_94x94")
        config_ag_folder_name = os.path.join("", "cfg")
        
    main(data_folder_name, config_ag_folder_name)