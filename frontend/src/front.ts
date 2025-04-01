interface Window {
    CredentialResponse: (response: any) => void;
}

let connected = false;

function handleConnection(input: boolean) {
    const connect = document.getElementById('connect');
    const profile = document.getElementById('profile');
    if (input && profile && connect) {
        connect.classList.add('hidden');
        profile.classList.toggle('hidden');
    } else if (profile && connect) {
        connect.classList.toggle('hidden');
        profile.classList.add('hidden');
    }
    connected = input;
}

window.CredentialResponse = async (credit: { credential: string }) => {
    try {
        const response = await fetch('http://localhost:3000/user-management/auth/google', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({credential: credit.credential})
        });
        if (!response.ok)
            console.error('Error: From UserManager returned an error');
        else {
            const reply = await response.json();
            if (reply.valid) {
                const container = document.getElementById('content') as HTMLElement;
                localStorage.setItem('token', reply.token);
                const res = await fetch('/part/connected');
                const newElement = document.createElement('div');
                newElement.className = 'tag';
                if (reply.nickName) {
                    console.log("UserGoogle:", reply.nickName);
                    localStorage.setItem('nickName', reply.nickName);
                    const nickName = localStorage.getItem('nickName');
                    const nameSpan = document.getElementById('nickName') as HTMLSpanElement;
                    nameSpan.textContent = reply.nickName;
                    console.log("Welcome", nickName);
                    const avatarImg = document.getElementById('avatar') as HTMLImageElement;
                    if (reply.avatar)
                        avatarImg.src = reply.avatar;
                    else
                        avatarImg.src = '../login.png';
                }
                if (!res.ok)
                    throw Error("Page not found: element missing.");
                const html = await res.text();
                if (html.includes(container.innerHTML))
                    return;
                container.innerHTML = '';
                newElement.innerHTML = html;
                container.appendChild(newElement);
                handleConnection(true);
            }
            console.log('Success:', reply);
        }
    }
    catch(error) {
        console.error('Error:', error);
    }
}


document.addEventListener("DOMContentLoaded", () => {
    // Constant button on the Single Page Application
    const aboutBtn = document.getElementById("about")!;
    const contactBtn = document.getElementById("contact")!;
    const Ping = document.getElementById("Ping")!;

    aboutBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/about"));
    contactBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/contact"));
    Ping.addEventListener("click", (event: MouseEvent) => navigate(event, "/pong"));

    // For Client Connection
    const connectBtn = document.getElementById('connect');
    const profilBtn = document.getElementById('profile');
    if (profilBtn && connected)
        profilBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/profile"));
    if (connectBtn && !connected)
        connectBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/connect"));
});

//
// const connect = document.getElementById('connect');
// const profil = document.getElementById('profil');
//
// if (Client.connected) {
//     connect.classList.add('hidden');
//     profil.classList.toggle('hidden');
//     const profil = document.getElementById("profil");
//     if (profil)
//         profil.addEventListener("click", (event: MouseEvent) => navigate(event, "/profil"));
//     const profile = document.getElementById("ddProfil")!;
//     const logout = document.getElementById("ddLogout")!;
//     profile.addEventListener("click", (event: MouseEvent) => navigate(event, "/profile"));
//     logout.addEventListener("click", (event: MouseEvent) => navigate(event, "/logout"));
// } else {
//     const connect = document.getElementById("connect");
//     if (connect)
//         connect.addEventListener("click", (event: MouseEvent) => navigate(event, "/connect"));
// }
// // connect.classList.add('hidden');

function navigate(event: MouseEvent, path: string): void {
    event.preventDefault();
    window.history.pushState({}, "", path);
    loadPart(path);
}

async function loadPart(page: string): Promise<void> {
    const container = document.getElementById('content') as HTMLElement;
    if (!page || page === "/") {
        container.innerHTML = '';
        container.innerHTML = "<p>Choisissez une option pour charger le contenu.</p>";
        return;
    }
    try {
        await insert_tag(`part${page}`);
        if (page === "/register") {
            const button = document.getElementById("registerButton")!;
            if (button)
                register(container, button);
        }
        if (page === "/login") {
            const button = document.getElementById("loginButton")!;
            if (button)
                login(container, button);
        }
        if (page === "/profile") {
            const nickName = document.getElementById("profileNickName")!;
            const email = document.getElementById("profileEmail")!;
            if (email && nickName)
                profile(container, nickName, email);
        }
        if (page === "/logout") {
            handleConnection(false);
            const avatar = document.getElementById("avatar") as HTMLImageElement;
            if (avatar)
                avatar.src = "../logout.png";
            const nickName = document.getElementById("nickName") as HTMLSpanElement;
            if (nickName)
                nickName.textContent = '';
        }
    } catch (error) {
        console.error(error);
        container.innerHTML = '';
        container.innerHTML = "<h2>404 - Page non trouvée</h2>";
    }
}

async function insert_tag(url: string): Promise<void>{
    const container = document.getElementById('content') as HTMLElement;
    if (url === "part/pong") {
        const existingScript = document.querySelector('script[src="/static/dist/pong.js"]');
        if (existingScript)
            return ;
    }
    const res = await fetch(url);
    const newElement = document.createElement('div');
    newElement.className = 'tag';
    if (!res.ok)
        throw Error("Page not found: element missing.");
    const html = await res.text();
    if (container.innerHTML.includes(html))
        return;
    if (html.includes(container.innerHTML))
        return;
    container.innerHTML = '';
    if (url === "part/pong") {
        if (!document.querySelector('script[src="/static/dist/pong.js"]')) {
            const script = document.createElement('script');
            script.src = "/static/dist/pong.js";
            document.body.appendChild(script);
        }
    } else {
        const existingScript = document.querySelector('script[src="/static/dist/pong.js"]');
        if (existingScript)
            existingScript.remove();
    }
    if (url === "part/login" || url === "part/connect") {
        const script = document.createElement('script');
        script.src = "https://accounts.google.com/gsi/client";
        script.async = true;
        script.defer = true;
        container.appendChild(script);
    } else {
        const googleID = document.getElementById('googleidentityservice');
        const googlemeta = document.querySelector('meta[http-equiv="origin-trial"]');
        if (googlemeta)
            googlemeta.remove();
        if (googleID)
            googleID.remove();
    }
    newElement.innerHTML = html;
    container.appendChild(newElement);
}

function register(container: HTMLElement, button: HTMLElement): void {
    button.addEventListener("click", async () => {
        const myForm = document.getElementById("myForm") as HTMLFormElement;
        const formData = new FormData(myForm);
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>);
        const response = await fetch('http://localhost:3000/user-management/sign-up', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (result.token) {
            localStorage.setItem('token', result.token);
            const res = await fetch('/part/connected');
            const newElement = document.createElement('div');
            newElement.className = 'tag';
            if (!res.ok)
                throw Error("Page not found: element missing.");
            const html = await res.text();
            if (html.includes(container.innerHTML))
                return;
            container.innerHTML = '';
            newElement.innerHTML = html;
            container.appendChild(newElement);
            handleConnection(true);
        } else {
            const errors = result.json;
            validateRegister(errors.json);
        }
    });
}

function login(container: HTMLElement, button: HTMLElement): void {
    button.addEventListener("click", async () => {
        const myForm = document.getElementById("myForm") as HTMLFormElement;
        const formData = new FormData(myForm);
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>);
        const response = await fetch('http://localhost:3000/user-management/sign-in', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (response.ok) {
            localStorage.setItem('token', result.token);
            localStorage.setItem('nickName', result.nickName);
            // localStorage.setItem('avatar', result.avatar);
            const res = await fetch('/part/connected');
            const newElement = document.createElement('div');
            newElement.className = 'tag';
            if (!res.ok)
                throw Error("Page not found: element missing.");
            const html = await res.text();
            if (html.includes(container.innerHTML))
                return;
            container.innerHTML = '';
            newElement.innerHTML = html;
            container.appendChild(newElement);
            handleConnection(true);
        } else {
            const loginError = document.getElementById("LoginError") as HTMLSpanElement;
            // if (!loginError.classList.contains("hidden"))
            loginError.textContent = result?.error ?? "An error occurred";
            loginError.classList.remove("hidden");
        }
    });
}

async function profile(container: HTMLElement, nickName: HTMLElement, email: HTMLElement) {
    try {
        const token = localStorage.getItem('token');
        if (!token) {
            console.error('token missing');
            return;
        }
        const response = await fetch('http://localhost:3000/user-management/getProfile', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token
            }
        })
        const data = await response.json();
        if (response.ok) {
            nickName.innerText = data.nickName;
            email.innerText = data.email;
        }
    }
    catch (err) {
        console.log(err)
    }
}

function validateRegister(result: { nickName: string; email: string; password: string}): void {
    const nickNameErrorMin = document.getElementById("nickNameErrorMin") as HTMLSpanElement;
    const emailError = document.getElementById("emailError") as HTMLSpanElement;
    const passwordError = document.getElementById("passwordError") as HTMLSpanElement;
    if (result.nickName)
        nickNameErrorMin.classList.remove("hidden");
    else {
        if (!nickNameErrorMin.classList.contains("hidden")) {
            nickNameErrorMin.classList.add("hidden");
        }
    }
    if (result.email)
        emailError.classList.remove("hidden");
    else {
        if (!emailError.classList.contains("hidden")) {
            emailError.classList.add("hidden");
        }
    }
    if (result.password)
        passwordError.classList.remove("hidden");
    else {
        if (!passwordError.classList.contains("hidden")) {
            passwordError.classList.add("hidden");
        }
    }
}