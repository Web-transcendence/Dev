# Project Overview
## Choice of Modules

### • Web
◦ Major module: Use a Framework to build the backend.🟩

◦ Minor module: Use a framework or a toolkit to build the frontend.🟩

◦ Minor module: Use a database for the backend.🟩

~~◦ Major module: Store the score of a tournament in the Blockchain.~~

### • User Management

◦ Major module: Standard user management, authentication, users across tournaments.🟥

◦ Major module: Implementing a remote authentication.🟩


### • Gameplay and user experience

◦ Major module: Remote players🟪

~~◦ Major module: Multiplayers (more than 2 in the same game).~~

◦ Major module: Add Another Game with User History and Matchmaking.🟪

◦ Minor module: Game Customization Options.🟪


~~◦ Major module: Live chat.~~


### • AI-Algo

◦ Major module: Introduce an AI Opponent.⬛

◦ Minor module: User and Game Stats Dashboards⬛

### • Cybersecurity

~~◦ Major module: Implement WAF/ModSecurity with Hardened Configuration and HashiCorp Vault for Secrets Management.~~

~~◦ Minor module: GDPR Compliance Options with User Anonymization, Local Data Management, and Account Deletion.~~

◦ Major module: Implement Two-Factor Authentication (2FA) and JWT.🟩

### • Devops

~~◦ Major module: Infrastructure Setup for Log Management.~~

~~◦ Minor module: Monitoring system.~~

◦ Major module: Designing the Backend as Microservices.🟥
### • Graphics

~~◦ Major module: Use of advanced 3D techniques.~~

### • Accessibility

~~◦ Minor module: Support on all devices.~~

~~◦ Minor module: Expanding Browser Compatibility.~~

~~◦ Minor module: Multiple language supports.~~

~~◦ Minor module: Add accessibility for Visually Impaired Users.~~

~~◦ Minor module: Server-Side Rendering (SSR) Integration.~~

### • Server-Side Pong

◦ Major module: Replacing Basic Pong with Server-Side Pong and Implementing an API.⬛🟪


~~◦ Major module: Enabling Pong Gameplay via CLI against Web Users with API Integration.~~

## Proccessus
  First Steps
Mise en place de la structure des dossiers et des docker puis du docker compose.
  Proxy
Connexion par un proxy du site web, format html. 
  
  - Command pour rentrer dans un container
```angular2html
docker exec -it <name> sh
```
RUN server without docker 
```angular2html
npm run dev
```

## Creation de Node.js
Faire une base en ECMAScript Modules (ESM)

