# ADR-002 Architecture Refinement
**Status:** Accepted

**Authors:**

- Mithran Mohanraj

## Context

For P1, we need to begin the creation of concrete UML diagrams describing the behavior and business logic of LlamaSearch.

## Decision

In order to give the team a clearer picture of LlamaSearch's business logic, I designed the following UML diagrams. I propose that we loosely follow the implementation they describe.

### Use Case Digagram

![](../assets/specialized-use-case.svg)

### Sequence Diagram
```mermaid
sequenceDiagram
    participant U as User
    participant UI as User Interface
    participant C as Crawler
    participant CE as Content Extractor
    participant P as Text Preprocessor
    participant VG as Vector Generator
    participant VDB as Vector Database
    participant R as Retrieval
    participant T as Trustworthiness Service
    participant LLM as Llama Model
    
    %% Search Input Flow
    rect rgb(200, 220, 240)
    Note over U,LLM: Search Input Flow
    U->>UI: Enter website URL
    UI->>C: Start crawling website
    C->>C: Crawl pages
    C->>CE: Send crawled pages
    CE->>CE: Extract content via Jina API
    CE->>P: Send extracted content
    P->>P: Convert to markdown
    P->>VG: Send processed text
    VG->>VG: Generate embeddings
    VG->>VDB: Store vectors
    VDB-->>UI: Confirm indexing complete
    UI-->>U: Enable chat interface
    end

    %% Chat Input Flow
    rect rgb(240, 220, 200)
    Note over U,LLM: Chat Input Flow
    U->>UI: Send chat message
    UI->>VG: Convert message to vector
    VG->>VDB: Perform similarity search
    VDB-->>R: Return top-K similar chunks
    R->>T: Get trustworthiness scores
    T-->>R: Return scores
    R->>LLM: Send query + relevant chunks
    LLM-->>UI: Generate response
    UI->>UI: Display response with citations
    UI-->>U: Show response, citations, and scores
    end
```

### Activity Diagram
```mermaid
flowchart TD
    Start([Start]) --> A{User Input Type?}
    
    %% Search Flow
    A -->|Search| B[Enter Website URL]
    B --> C[Validate URL]
    C --> D{Valid URL?}
    D -->|No| B
    D -->|Yes| E[Start Crawling]
    
    E --> F[Extract Content]
    F --> G[Convert to Markdown]
    
    G --> H[Generate Vectors]
    H --> I[Store in Database]
    I --> J[Enable Chat Interface]
    
    %% Chat Flow
    A -->|Chat| K{Is Search Complete?}
    K -->|No| L[Show Error Message]
    K -->|Yes| M[Process Query]
    
    M --> N[Convert to Vector]
    N --> O[Search Similar Chunks]
    
    O --> P[Get Trustworthiness Scores]
    P --> Q[Generate Response]
    
    Q --> R[Format Response]
    R --> S[Display Results]
    
    %% Feedback Flow
    S --> T{First Chat?}
    T -->|Yes| U[Request Feedback]
    T -->|No| V[Continue]
    
    %% Styling
    classDef process fill:#f9f,stroke:#333,s troke-width:2px;
    classDef decision fill:#ff9,stroke:#333,stroke-width:2px;
    classDef start fill:#9f9,stroke:#333,stroke-width:2px;
    
    class A,D,K,T decision;
    class Start start;
    class B,C,E,F,G,H,I,J,M,N,O,P,Q,R,S,U,V process;
```

### Class Diagram
```mermaid
classDiagram
    class SearchManager {
        -website: string
        -isIndexingComplete: boolean
        +startSearch(url: string)
        +getIndexingStatus(): boolean
    }

    class Crawler {
        -maxDepth: int
        -visitedUrls: List<string>
        +crawl(url: string): List<Page>
        -isValidUrl(url: string): boolean
    }

    class ContentExtractor {
        -jinaApiKey: string
        +extract(pages: List<Page>): List<Content>
        -cleanContent(raw: string): string
    }

    class TextPreprocessor {
        +processContent(content: Content): Document
        -convertToMarkdown(text: string): string
    }

    class VectorGenerator {
        -modelName: string
        +generateEmbeddings(doc: Document): Vector
        -normalizeVector(vec: Vector): Vector
    }

    class VectorDatabase {
        -vectors: List<Vector>
        -metadata: Map<string, string>
        +store(vector: Vector, metadata: Map)
        +search(query: Vector, k: int): List<Result>
    }

    class ChatManager {
        -context: List<Message>
        +processQuery(query: string): Response
        -formatResponse(raw: string): Response
    }

    class TrustworthinessService {
        -scoreCache: Map<string, float>
        +getScore(url: string): float
        -updateCache(url: string, score: float)
    }

    class LlamaModel {
        -modelConfig: Config
        +generate(prompt: string): string
        -formatPrompt(context: string, query: string): string
    }

    class UserInterface {
        -currentState: State
        +handleSearchInput(url: string)
        +handleChatInput(query: string)
        +displayResponse(response: Response)
    }

    UserInterface --> SearchManager
    UserInterface --> ChatManager
    SearchManager --> Crawler
    Crawler --> ContentExtractor
    ContentExtractor --> TextPreprocessor
    TextPreprocessor --> VectorGenerator
    VectorGenerator --> VectorDatabase
    ChatManager --> VectorDatabase
    ChatManager --> TrustworthinessService
    ChatManager --> LlamaModel
```

## Explanation

These diagrams were derived from the initial architecture diagram in [ADR-001](./001-initial-architecture.md). They are not comprehensive, and will need revisions in future ADRs.

## Review Triggers
- We begin developing a feature according to a diagram and discover it is not feasible
- We find flaws in the design described by the diagrams
