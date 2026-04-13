# Test Prediction Analysis Report
This report analyzes the 'Actual vs Predicted' ordering for sample documents across different embedding methods.

### Embedding: 1. Word2Vec (Mean Pooled)

#### Sample Document 1 (ID: 70)
- **Kendall's Tau Score**: -0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the paper presents the system developed by racai for the iswlt 2012 competition ... | we further present our attempts at creating a better controlled decoder than the... | ❌ |
| 2 | we describe the starting baseline phrasebased smt system  the experiments conduc... | the paper presents the system developed by racai for the iswlt 2012 competition ... | ❌ |
| 3 | we further present our attempts at creating a better controlled decoder than the... | we describe the starting baseline phrasebased smt system  the experiments conduc... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 3 as start.

#### Sample Document 2 (ID: 32)
- **Kendall's Tau Score**: 0.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper proposes a way of augmenting bilingual terminologies by using a gener... | using existing bilingual terminologies  the method generates potential  bilingua... | ❌ |
| 2 | using existing bilingual terminologies  the method generates potential  bilingua... | unlike most existing bilingual term extraction methods  which use parallel or co... | ❌ |
| 3 | unlike most existing bilingual term extraction methods  which use parallel or co... | experiments using japanese-english terminologies of five domains show that the m... | ❌ |
| 4 | experiments using japanese-english terminologies of five domains show that the m... | this paper proposes a way of augmenting bilingual terminologies by using a gener... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 2 as start.

#### Sample Document 3 (ID: 30)
- **Kendall's Tau Score**: -0.4667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | in this paper  we present an approach to rich morphology prediction using a para... | then  we predict six morphological features of the verb and generate inflected v... | ❌ |
| 2 | we focus on the verb conjugation as the most important and problematic phenomeno... | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | ❌ |
| 3 | we define a set of linguistic features using both english and persian linguistic... | in our experiments  we generate verb form with the most common feature values as... | ❌ |
| 4 | then  we predict six morphological features of the verb and generate inflected v... | in this paper  we present an approach to rich morphology prediction using a para... | ❌ |
| 5 | in our experiments  we generate verb form with the most common feature values as... | we define a set of linguistic features using both english and persian linguistic... | ❌ |
| 6 | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | we focus on the verb conjugation as the most important and problematic phenomeno... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 4 as start.

#### Sample Document 4 (ID: 96)
- **Kendall's Tau Score**: -0.0667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | dialogue act annotation aids understanding of interaction structure  and also in... | these categories are not mutually exclusive  many service encounters include soc... | ❌ |
| 2 | while many dialogues can be described as task-based or instrumental  others are ... | while many dialogues can be described as task-based or instrumental  others are ... | ✅ |
| 3 | these categories are not mutually exclusive  many service encounters include soc... | in this paper we briefly describe social or casual talk  review how current dial... | ❌ |
| 4 | much research on dialogue and particularly on description of dialogue acts for u... | however  attention has been focusing on social aspects of spoken and text intera... | ❌ |
| 5 | however  attention has been focusing on social aspects of spoken and text intera... | dialogue act annotation aids understanding of interaction structure  and also in... | ❌ |
| 6 | in this paper we briefly describe social or casual talk  review how current dial... | much research on dialogue and particularly on description of dialogue acts for u... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 3 as start.

#### Sample Document 5 (ID: 107)
- **Kendall's Tau Score**: 0.6667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we propose a static relation extraction task to complement biomedical informatio... | we argue that static relations such as part-whole are implicitly involved in man... | ❌ |
| 2 | we argue that static relations such as part-whole are implicitly involved in man... | we propose a static relation extraction task to complement biomedical informatio... | ❌ |
| 3 | we further identify a specific static relation extraction task motivated by the ... | we further identify a specific static relation extraction task motivated by the ... | ✅ |
| 4 | the task setting and corpus can serve to support several forms of domain informa... | the task setting and corpus can serve to support several forms of domain informa... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 2 as start. Correctly identified the concluding sentence. Contains 1 adjacent sentence swap(s).


---

### Embedding: 2. Word2Vec (TF-IDF Weighted)

#### Sample Document 1 (ID: 70)
- **Kendall's Tau Score**: -0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the paper presents the system developed by racai for the iswlt 2012 competition ... | we describe the starting baseline phrasebased smt system  the experiments conduc... | ❌ |
| 2 | we describe the starting baseline phrasebased smt system  the experiments conduc... | we further present our attempts at creating a better controlled decoder than the... | ❌ |
| 3 | we further present our attempts at creating a better controlled decoder than the... | the paper presents the system developed by racai for the iswlt 2012 competition ... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 2 as start.

#### Sample Document 2 (ID: 32)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper proposes a way of augmenting bilingual terminologies by using a gener... | this paper proposes a way of augmenting bilingual terminologies by using a gener... | ✅ |
| 2 | using existing bilingual terminologies  the method generates potential  bilingua... | unlike most existing bilingual term extraction methods  which use parallel or co... | ❌ |
| 3 | unlike most existing bilingual term extraction methods  which use parallel or co... | experiments using japanese-english terminologies of five domains show that the m... | ❌ |
| 4 | experiments using japanese-english terminologies of five domains show that the m... | using existing bilingual terminologies  the method generates potential  bilingua... | ❌ |

**Analysis**: Correctly identified the opening sentence.

#### Sample Document 3 (ID: 30)
- **Kendall's Tau Score**: 0.2000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | in this paper  we present an approach to rich morphology prediction using a para... | in this paper  we present an approach to rich morphology prediction using a para... | ✅ |
| 2 | we focus on the verb conjugation as the most important and problematic phenomeno... | in our experiments  we generate verb form with the most common feature values as... | ❌ |
| 3 | we define a set of linguistic features using both english and persian linguistic... | we define a set of linguistic features using both english and persian linguistic... | ✅ |
| 4 | then  we predict six morphological features of the verb and generate inflected v... | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | ❌ |
| 5 | in our experiments  we generate verb form with the most common feature values as... | we focus on the verb conjugation as the most important and problematic phenomeno... | ❌ |
| 6 | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | then  we predict six morphological features of the verb and generate inflected v... | ❌ |

**Analysis**: Correctly identified the opening sentence.

#### Sample Document 4 (ID: 96)
- **Kendall's Tau Score**: -0.0667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | dialogue act annotation aids understanding of interaction structure  and also in... | much research on dialogue and particularly on description of dialogue acts for u... | ❌ |
| 2 | while many dialogues can be described as task-based or instrumental  others are ... | dialogue act annotation aids understanding of interaction structure  and also in... | ❌ |
| 3 | these categories are not mutually exclusive  many service encounters include soc... | in this paper we briefly describe social or casual talk  review how current dial... | ❌ |
| 4 | much research on dialogue and particularly on description of dialogue acts for u... | however  attention has been focusing on social aspects of spoken and text intera... | ❌ |
| 5 | however  attention has been focusing on social aspects of spoken and text intera... | while many dialogues can be described as task-based or instrumental  others are ... | ❌ |
| 6 | in this paper we briefly describe social or casual talk  review how current dial... | these categories are not mutually exclusive  many service encounters include soc... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 4 as start.

#### Sample Document 5 (ID: 107)
- **Kendall's Tau Score**: 0.6667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we propose a static relation extraction task to complement biomedical informatio... | we propose a static relation extraction task to complement biomedical informatio... | ✅ |
| 2 | we argue that static relations such as part-whole are implicitly involved in man... | we further identify a specific static relation extraction task motivated by the ... | ❌ |
| 3 | we further identify a specific static relation extraction task motivated by the ... | we argue that static relations such as part-whole are implicitly involved in man... | ❌ |
| 4 | the task setting and corpus can serve to support several forms of domain informa... | the task setting and corpus can serve to support several forms of domain informa... | ✅ |

**Analysis**: Correctly identified the opening sentence. Correctly identified the concluding sentence. Contains 1 adjacent sentence swap(s).


---

### Embedding: 3. TF-IDF (Sparse Vector)

#### Sample Document 1 (ID: 70)
- **Kendall's Tau Score**: -0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the paper presents the system developed by racai for the iswlt 2012 competition ... | we describe the starting baseline phrasebased smt system  the experiments conduc... | ❌ |
| 2 | we describe the starting baseline phrasebased smt system  the experiments conduc... | we further present our attempts at creating a better controlled decoder than the... | ❌ |
| 3 | we further present our attempts at creating a better controlled decoder than the... | the paper presents the system developed by racai for the iswlt 2012 competition ... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 2 as start.

#### Sample Document 2 (ID: 32)
- **Kendall's Tau Score**: -0.6667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper proposes a way of augmenting bilingual terminologies by using a gener... | experiments using japanese-english terminologies of five domains show that the m... | ❌ |
| 2 | using existing bilingual terminologies  the method generates potential  bilingua... | unlike most existing bilingual term extraction methods  which use parallel or co... | ❌ |
| 3 | unlike most existing bilingual term extraction methods  which use parallel or co... | this paper proposes a way of augmenting bilingual terminologies by using a gener... | ❌ |
| 4 | experiments using japanese-english terminologies of five domains show that the m... | using existing bilingual terminologies  the method generates potential  bilingua... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 4 as start.

#### Sample Document 3 (ID: 30)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | in this paper  we present an approach to rich morphology prediction using a para... | in our experiments  we generate verb form with the most common feature values as... | ❌ |
| 2 | we focus on the verb conjugation as the most important and problematic phenomeno... | we focus on the verb conjugation as the most important and problematic phenomeno... | ✅ |
| 3 | we define a set of linguistic features using both english and persian linguistic... | in this paper  we present an approach to rich morphology prediction using a para... | ❌ |
| 4 | then  we predict six morphological features of the verb and generate inflected v... | we define a set of linguistic features using both english and persian linguistic... | ❌ |
| 5 | in our experiments  we generate verb form with the most common feature values as... | then  we predict six morphological features of the verb and generate inflected v... | ❌ |
| 6 | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 5 as start. Correctly identified the concluding sentence.

#### Sample Document 4 (ID: 96)
- **Kendall's Tau Score**: 0.2000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | dialogue act annotation aids understanding of interaction structure  and also in... | much research on dialogue and particularly on description of dialogue acts for u... | ❌ |
| 2 | while many dialogues can be described as task-based or instrumental  others are ... | while many dialogues can be described as task-based or instrumental  others are ... | ✅ |
| 3 | these categories are not mutually exclusive  many service encounters include soc... | dialogue act annotation aids understanding of interaction structure  and also in... | ❌ |
| 4 | much research on dialogue and particularly on description of dialogue acts for u... | however  attention has been focusing on social aspects of spoken and text intera... | ❌ |
| 5 | however  attention has been focusing on social aspects of spoken and text intera... | in this paper we briefly describe social or casual talk  review how current dial... | ❌ |
| 6 | in this paper we briefly describe social or casual talk  review how current dial... | these categories are not mutually exclusive  many service encounters include soc... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 4 as start.

#### Sample Document 5 (ID: 107)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we propose a static relation extraction task to complement biomedical informatio... | we propose a static relation extraction task to complement biomedical informatio... | ✅ |
| 2 | we argue that static relations such as part-whole are implicitly involved in man... | the task setting and corpus can serve to support several forms of domain informa... | ❌ |
| 3 | we further identify a specific static relation extraction task motivated by the ... | we argue that static relations such as part-whole are implicitly involved in man... | ❌ |
| 4 | the task setting and corpus can serve to support several forms of domain informa... | we further identify a specific static relation extraction task motivated by the ... | ❌ |

**Analysis**: Correctly identified the opening sentence.


---

### Embedding: 4. Contextual Token (DistilBERT)

#### Sample Document 1 (ID: 70)
- **Kendall's Tau Score**: 1.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the paper presents the system developed by racai for the iswlt 2012 competition ... | the paper presents the system developed by racai for the iswlt 2012 competition ... | ✅ |
| 2 | we describe the starting baseline phrasebased smt system  the experiments conduc... | we describe the starting baseline phrasebased smt system  the experiments conduc... | ✅ |
| 3 | we further present our attempts at creating a better controlled decoder than the... | we further present our attempts at creating a better controlled decoder than the... | ✅ |

**Analysis**: Perfect ordering.

#### Sample Document 2 (ID: 32)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper proposes a way of augmenting bilingual terminologies by using a gener... | unlike most existing bilingual term extraction methods  which use parallel or co... | ❌ |
| 2 | using existing bilingual terminologies  the method generates potential  bilingua... | this paper proposes a way of augmenting bilingual terminologies by using a gener... | ❌ |
| 3 | unlike most existing bilingual term extraction methods  which use parallel or co... | using existing bilingual terminologies  the method generates potential  bilingua... | ❌ |
| 4 | experiments using japanese-english terminologies of five domains show that the m... | experiments using japanese-english terminologies of five domains show that the m... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 3 as start. Correctly identified the concluding sentence.

#### Sample Document 3 (ID: 30)
- **Kendall's Tau Score**: 1.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | in this paper  we present an approach to rich morphology prediction using a para... | in this paper  we present an approach to rich morphology prediction using a para... | ✅ |
| 2 | we focus on the verb conjugation as the most important and problematic phenomeno... | we focus on the verb conjugation as the most important and problematic phenomeno... | ✅ |
| 3 | we define a set of linguistic features using both english and persian linguistic... | we define a set of linguistic features using both english and persian linguistic... | ✅ |
| 4 | then  we predict six morphological features of the verb and generate inflected v... | then  we predict six morphological features of the verb and generate inflected v... | ✅ |
| 5 | in our experiments  we generate verb form with the most common feature values as... | in our experiments  we generate verb form with the most common feature values as... | ✅ |
| 6 | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | ✅ |

**Analysis**: Correctly identified the opening sentence. Correctly identified the concluding sentence.

#### Sample Document 4 (ID: 96)
- **Kendall's Tau Score**: 0.0667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | dialogue act annotation aids understanding of interaction structure  and also in... | much research on dialogue and particularly on description of dialogue acts for u... | ❌ |
| 2 | while many dialogues can be described as task-based or instrumental  others are ... | while many dialogues can be described as task-based or instrumental  others are ... | ✅ |
| 3 | these categories are not mutually exclusive  many service encounters include soc... | dialogue act annotation aids understanding of interaction structure  and also in... | ❌ |
| 4 | much research on dialogue and particularly on description of dialogue acts for u... | in this paper we briefly describe social or casual talk  review how current dial... | ❌ |
| 5 | however  attention has been focusing on social aspects of spoken and text intera... | however  attention has been focusing on social aspects of spoken and text intera... | ✅ |
| 6 | in this paper we briefly describe social or casual talk  review how current dial... | these categories are not mutually exclusive  many service encounters include soc... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 4 as start.

#### Sample Document 5 (ID: 107)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we propose a static relation extraction task to complement biomedical informatio... | we propose a static relation extraction task to complement biomedical informatio... | ✅ |
| 2 | we argue that static relations such as part-whole are implicitly involved in man... | the task setting and corpus can serve to support several forms of domain informa... | ❌ |
| 3 | we further identify a specific static relation extraction task motivated by the ... | we argue that static relations such as part-whole are implicitly involved in man... | ❌ |
| 4 | the task setting and corpus can serve to support several forms of domain informa... | we further identify a specific static relation extraction task motivated by the ... | ❌ |

**Analysis**: Correctly identified the opening sentence.


---

### Embedding: 5. Sequence Domain (SBERT)

#### Sample Document 1 (ID: 70)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the paper presents the system developed by racai for the iswlt 2012 competition ... | the paper presents the system developed by racai for the iswlt 2012 competition ... | ✅ |
| 2 | we describe the starting baseline phrasebased smt system  the experiments conduc... | we further present our attempts at creating a better controlled decoder than the... | ❌ |
| 3 | we further present our attempts at creating a better controlled decoder than the... | we describe the starting baseline phrasebased smt system  the experiments conduc... | ❌ |

**Analysis**: Correctly identified the opening sentence. Contains 1 adjacent sentence swap(s).

#### Sample Document 2 (ID: 32)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper proposes a way of augmenting bilingual terminologies by using a gener... | using existing bilingual terminologies  the method generates potential  bilingua... | ❌ |
| 2 | using existing bilingual terminologies  the method generates potential  bilingua... | unlike most existing bilingual term extraction methods  which use parallel or co... | ❌ |
| 3 | unlike most existing bilingual term extraction methods  which use parallel or co... | this paper proposes a way of augmenting bilingual terminologies by using a gener... | ❌ |
| 4 | experiments using japanese-english terminologies of five domains show that the m... | experiments using japanese-english terminologies of five domains show that the m... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 2 as start. Correctly identified the concluding sentence.

#### Sample Document 3 (ID: 30)
- **Kendall's Tau Score**: 0.7333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | in this paper  we present an approach to rich morphology prediction using a para... | in this paper  we present an approach to rich morphology prediction using a para... | ✅ |
| 2 | we focus on the verb conjugation as the most important and problematic phenomeno... | we focus on the verb conjugation as the most important and problematic phenomeno... | ✅ |
| 3 | we define a set of linguistic features using both english and persian linguistic... | then  we predict six morphological features of the verb and generate inflected v... | ❌ |
| 4 | then  we predict six morphological features of the verb and generate inflected v... | we define a set of linguistic features using both english and persian linguistic... | ❌ |
| 5 | in our experiments  we generate verb form with the most common feature values as... | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | ❌ |
| 6 | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | in our experiments  we generate verb form with the most common feature values as... | ❌ |

**Analysis**: Correctly identified the opening sentence. Contains 2 adjacent sentence swap(s).

#### Sample Document 4 (ID: 96)
- **Kendall's Tau Score**: 0.4667
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | dialogue act annotation aids understanding of interaction structure  and also in... | dialogue act annotation aids understanding of interaction structure  and also in... | ✅ |
| 2 | while many dialogues can be described as task-based or instrumental  others are ... | much research on dialogue and particularly on description of dialogue acts for u... | ❌ |
| 3 | these categories are not mutually exclusive  many service encounters include soc... | while many dialogues can be described as task-based or instrumental  others are ... | ❌ |
| 4 | much research on dialogue and particularly on description of dialogue acts for u... | however  attention has been focusing on social aspects of spoken and text intera... | ❌ |
| 5 | however  attention has been focusing on social aspects of spoken and text intera... | in this paper we briefly describe social or casual talk  review how current dial... | ❌ |
| 6 | in this paper we briefly describe social or casual talk  review how current dial... | these categories are not mutually exclusive  many service encounters include soc... | ❌ |

**Analysis**: Correctly identified the opening sentence.

#### Sample Document 5 (ID: 107)
- **Kendall's Tau Score**: 0.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we propose a static relation extraction task to complement biomedical informatio... | we further identify a specific static relation extraction task motivated by the ... | ❌ |
| 2 | we argue that static relations such as part-whole are implicitly involved in man... | we propose a static relation extraction task to complement biomedical informatio... | ❌ |
| 3 | we further identify a specific static relation extraction task motivated by the ... | the task setting and corpus can serve to support several forms of domain informa... | ❌ |
| 4 | the task setting and corpus can serve to support several forms of domain informa... | we argue that static relations such as part-whole are implicitly involved in man... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 3 as start.


---

### Embedding: 6. Fine-Tuned Token (DistilBERT [CLS])

#### Sample Document 1 (ID: 70)
- **Kendall's Tau Score**: 1.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | the paper presents the system developed by racai for the iswlt 2012 competition ... | the paper presents the system developed by racai for the iswlt 2012 competition ... | ✅ |
| 2 | we describe the starting baseline phrasebased smt system  the experiments conduc... | we describe the starting baseline phrasebased smt system  the experiments conduc... | ✅ |
| 3 | we further present our attempts at creating a better controlled decoder than the... | we further present our attempts at creating a better controlled decoder than the... | ✅ |

**Analysis**: Perfect ordering.

#### Sample Document 2 (ID: 32)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | this paper proposes a way of augmenting bilingual terminologies by using a gener... | unlike most existing bilingual term extraction methods  which use parallel or co... | ❌ |
| 2 | using existing bilingual terminologies  the method generates potential  bilingua... | this paper proposes a way of augmenting bilingual terminologies by using a gener... | ❌ |
| 3 | unlike most existing bilingual term extraction methods  which use parallel or co... | using existing bilingual terminologies  the method generates potential  bilingua... | ❌ |
| 4 | experiments using japanese-english terminologies of five domains show that the m... | experiments using japanese-english terminologies of five domains show that the m... | ✅ |

**Analysis**: Incorrect opening: Predicted sentence 3 as start. Correctly identified the concluding sentence.

#### Sample Document 3 (ID: 30)
- **Kendall's Tau Score**: 1.0000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | in this paper  we present an approach to rich morphology prediction using a para... | in this paper  we present an approach to rich morphology prediction using a para... | ✅ |
| 2 | we focus on the verb conjugation as the most important and problematic phenomeno... | we focus on the verb conjugation as the most important and problematic phenomeno... | ✅ |
| 3 | we define a set of linguistic features using both english and persian linguistic... | we define a set of linguistic features using both english and persian linguistic... | ✅ |
| 4 | then  we predict six morphological features of the verb and generate inflected v... | then  we predict six morphological features of the verb and generate inflected v... | ✅ |
| 5 | in our experiments  we generate verb form with the most common feature values as... | in our experiments  we generate verb form with the most common feature values as... | ✅ |
| 6 | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | the results of our experiments show an improvement of almost 2 1  absolute bleu ... | ✅ |

**Analysis**: Correctly identified the opening sentence. Correctly identified the concluding sentence.

#### Sample Document 4 (ID: 96)
- **Kendall's Tau Score**: -0.2000
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | dialogue act annotation aids understanding of interaction structure  and also in... | much research on dialogue and particularly on description of dialogue acts for u... | ❌ |
| 2 | while many dialogues can be described as task-based or instrumental  others are ... | dialogue act annotation aids understanding of interaction structure  and also in... | ❌ |
| 3 | these categories are not mutually exclusive  many service encounters include soc... | in this paper we briefly describe social or casual talk  review how current dial... | ❌ |
| 4 | much research on dialogue and particularly on description of dialogue acts for u... | however  attention has been focusing on social aspects of spoken and text intera... | ❌ |
| 5 | however  attention has been focusing on social aspects of spoken and text intera... | these categories are not mutually exclusive  many service encounters include soc... | ❌ |
| 6 | in this paper we briefly describe social or casual talk  review how current dial... | while many dialogues can be described as task-based or instrumental  others are ... | ❌ |

**Analysis**: Incorrect opening: Predicted sentence 4 as start.

#### Sample Document 5 (ID: 107)
- **Kendall's Tau Score**: 0.3333
| Order | Actual Sentence | Predicted Sentence | Match? |
| :--- | :--- | :--- | :--- |
| 1 | we propose a static relation extraction task to complement biomedical informatio... | we propose a static relation extraction task to complement biomedical informatio... | ✅ |
| 2 | we argue that static relations such as part-whole are implicitly involved in man... | the task setting and corpus can serve to support several forms of domain informa... | ❌ |
| 3 | we further identify a specific static relation extraction task motivated by the ... | we argue that static relations such as part-whole are implicitly involved in man... | ❌ |
| 4 | the task setting and corpus can serve to support several forms of domain informa... | we further identify a specific static relation extraction task motivated by the ... | ❌ |

**Analysis**: Correctly identified the opening sentence.


---

