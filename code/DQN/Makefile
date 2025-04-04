# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/07/15 17:21:22 by thibaud           #+#    #+#              #
#    Updated: 2025/04/03 21:57:53 by thibaud          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

NAME		= myDQN

HEADER		=	hdrs/
SRCS_DIR	=	srcs/
OBJS_DIR	=	.objs/
DEPS_DIR	=	.deps/
SRCS		=	main.cpp \
				Agent.cpp \
				ExpReplay.cpp \
				Environment.cpp \
				Network.cpp \
				Layer.cpp \
				Neuron.cpp \
				Math.cpp

OBJS		= $(SRCS:%.cpp=$(OBJS_DIR)%.o)
DEPS		= $(SRCS:%.cpp=$(DEPS_DIR)%.d)

CC		= c++
CFLAGS	= -Wall -Wextra -Werror -g3 -I $(HEADER)
DFLAGS	= -MMD -MP -MT $@ -MF $(DEPS_DIR)$*.d
RM		= rm -rf

-include $(DEPS)

.DEFAULT_GOAL = all

all: $(NAME)

run: all
	./$(NAME)

$(NAME): $(OBJS)
	$(CC) $(CFLAGS) -o $(NAME) $(OBJS) && echo "compilation OK...\n"

$(OBJS_DIR):
	mkdir -p $(OBJS_DIR)
$(DEPS_DIR):
	mkdir -p $(DEPS_DIR)

$(OBJS): $(OBJS_DIR)%.o: $(SRCS_DIR)%.cpp | $(OBJS_DIR) $(DEPS_DIR)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@

clean:
		$(RM) $(OBJS_DIR) $(DEPS_DIR) && echo "Objects cleared...\n"

fclean: clean
		$(RM) $(NAME) && $(RM) $(DIR_BUILD) $(OBJS_DIR) $(DEPS_DIR) && echo "fully cleared...\n"

re: fclean
	$(MAKE) all

.PHONY: all clean fclean re
