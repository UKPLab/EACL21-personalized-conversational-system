# encoding:utf-8
import os
import re


class Parser(object):
    def __init__(self, persona_limit=None, set_relation=True):
        self.your_persona = []
        self.partner_persona = []
        self.personas = []
        self.conversation = []

        self.persona_limit = persona_limit
        self.set_relation = set_relation

    def parse(self, path):
        base_dir = os.path.dirname(__file__)
        f = open(os.path.join(base_dir, path), 'r', encoding='UTF-8')
        your_current = []
        partner_current = []
        conversation_current = []
        for i, line in enumerate(f.readlines()):
            if (not i == 0) and (line[0:2] == '1 '):
                # print('your_current', your_current)
                # print('partner_current', partner_current)
                # print('conversation_current', conversation_current)
                self.add_new(self.your_persona, your_current)
                self.add_new(self.partner_persona, partner_current)
                self.add_new(self.personas, your_current)
                self.add_new(self.personas, partner_current)
                self.conversation.append(conversation_current[:])
                your_current.clear()
                partner_current.clear()
                conversation_current.clear()
            if '\t' not in line:   # persona
                # line = line.replace(' don t ', ' do not ').replace(' doesn t ', ' does not ').replace(' i m ', ' i am ').replace(' ve ', ' have ').replace(' ll ', ' will ').replace(' aren t ', ' are not ').replace(' isn t ', ' is not ').replace(' didn t ', ' did not ').replace(' can t ', ' can not ').replace(' cannot ', ' can not ').replace(' couldn t ', ' could not ').replace(' haven t ', ' have not ').replace(' hadn t ', ' had not ').replace(' shouldn t ', ' should not ').replace(' i d ', ' i would ').replace(' he s ', ' he is ').replace(' she s ', ' she is ').replace(' it s ', ' it is ').replace(' s ', 's ').replace(' re ', ' are ')
                # line = line.replace(' don t.', ' do not.').replace(' doesn t.', ' does not.').replace(' i m.', ' i am.').replace(' ve.', ' have.').replace(' ll.', ' will.').replace(' aren t.', ' are not.').replace(' isn t.', ' is not.').replace(' didn t.', ' did not.').replace(' can t.', ' can not.').replace(' cannot.', ' can not.').replace(' couldn t.', ' could not.').replace(' haven t.', ' have not.').replace(' hadn t.', ' had not.').replace(' shouldn t.', ' should not.').replace(' i d.', ' i would.').replace(' he s.', ' he is.').replace(' she s.', ' she is.').replace(' it s.', ' it is.').replace(' s.', 's.').replace(' re.', ' are.')
                line = line.split()
                if line[1] == 'your' and line[2] == 'persona:':
                    your_current.append(' '.join(line[3:]))
                    # your_current.extend(self.tokenize(line[3:]))
                elif line[1] == 'partner\'s' and line[2] == 'persona:':
                    partner_current.append(' '.join(line[3:]))
            else:   # conversation
                conversations = line.split('\t\t')[0].split('\t')
                conversations[0] = conversations[0].split()
                conversations[1] = conversations[1].split()
                conversation_current.append(' '.join(conversations[0][1:]))
                conversation_current.append(' '.join(conversations[1]))
        else:
            self.add_new(self.your_persona, your_current)
            self.add_new(self.partner_persona, partner_current)
            self.add_new(self.personas, your_current)
            self.add_new(self.personas, partner_current)
            self.conversation.append(conversation_current[:])
        f.close()

    def tokenize(self, sentence):
        tokens = []
        for token in sentence:
            if re.findall(r"\w+", token) is not None:
                tokens.extend(re.findall(r"\w+", token))
        return tokens

    def add_new(self, dic, to_add):
        for e in dic:
            if self.set_relation is True:
                if set(e) >= set(to_add):
                    break
                elif set(e) < set(to_add):
                    dic.remove(e)
            elif self.set_relation is False:
                if set(e) == set(to_add):
                    break
        else:
            if self.persona_limit is None:
                dic.append(to_add[:])
            elif len(to_add) >= self.persona_limit:
                dic.append(to_add[:])
