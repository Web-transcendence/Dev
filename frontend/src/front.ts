interface Window {
    CredentialResponse: (response: any) => void;
}

window.CredentialResponse = async (credit: { credential: string }) => {
    console.log('Google response:', credit); // Vérifiez si ce log apparaît
    console.log('Google credential:', credit.credential); // Vérifiez si ce log apparaît
    try { // FETCH BASE
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
                // @ts-ignore
                // localStorage.setItem('credit', credit);
                const res = await fetch('/part/connected');
                const newElement = document.createElement('div');
                newElement.className = 'tag';
                if (reply.name) {
                    console.log("UserGoogle:", reply.name);
                    localStorage.setItem('name', reply.name); // Stockage

                    const username = localStorage.getItem('username'); // Récupération
                    const nameSpan = document.getElementById('username') as HTMLSpanElement;
                    nameSpan.textContent = username;
                    console.log("Welcome", username);
                    const avatarImg = document.getElementById('avatar') as HTMLImageElement;
                    // Récupérer l'avatar en utilisant l'API Google (exemple avec le jeton

                    if (reply.avatar)
                        avatarImg.src = reply.avatar;
                    else
                        avatarImg.src = '../login.png'; // Image par défaut si pas d'avatar
                }
                if (!res.ok)
                    throw Error("Page not found: element missing.");
                const html = await res.text();
                if (html.includes(container.innerHTML))
                    return;
                container.innerHTML = '';
                newElement.innerHTML = html;
                container.appendChild(newElement);
            }
            console.log('Success:', reply);
        }
    }
    catch(error) {
        console.error('Error:', error);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const aboutBtn = document.getElementById("about")!;
    const contactBtn = document.getElementById("contact")!;
    const registerBtn = document.getElementById("register")!;
    const loginBtn = document.getElementById("login")!;

    aboutBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/about"));
    contactBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/contact"));
    registerBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/register"));
    loginBtn.addEventListener("click", (event: MouseEvent) => navigate(event, "/login"));

    CheckForToken();
});

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
    } catch (error) {
        console.error(error);
        container.innerHTML = '';
        container.innerHTML = "<h2>404 - Page non trouvée</h2>";
    }
}

async function insert_tag(url: string): Promise<void>{
    const container = document.getElementById('content') as HTMLElement;
    const res = await fetch(url);
    const newElement = document.createElement('div');
    const scriptElement = document.createElement('script');
    newElement.className = 'tag';
    if (!res.ok)
        throw Error("Page not found: element missing.");
    const html = await res.text();
    if (html.includes(container.innerHTML))
        return;
    container.innerHTML = '';
    const script = document.createElement('script');
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;

    // Set the innerHTML of the new element to the fetched HTML
    newElement.innerHTML = html;

    // Append the script element and new content to the container
    container.appendChild(newElement);
    container.appendChild(script);

    console.log(script);
}

function register(container: HTMLElement, button: HTMLElement): void {
    button.addEventListener("click", async (event) => {
        const myForm = document.getElementById("myForm") as HTMLFormElement;
        const formData = new FormData(myForm);
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>);
        const response = await fetch('http://localhost:3000/user-management/sign-up', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        console.log(result);
        console.log(result.json);
        if (result.token) {
            localStorage.setItem('token', result.token);
            await CheckForToken();
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
        } else {
            const errors = result.json;
            validateRegister(errors.json);
        }
    });
}

function login(container: HTMLElement, button: HTMLElement): void {
    button.addEventListener("click", async (event) => {
        const myForm = document.getElementById("myForm") as HTMLFormElement;
        const formData = new FormData(myForm);
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>);
        console.log("DATA", data);
        const response = await fetch('http://localhost:3000/user-management/user-login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (result.valid) {
            localStorage.setItem('token', result.token);
            localStorage.setItem('name', result.name);
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
        } else {
            const loginError = document.getElementById("LoginError") as HTMLSpanElement;
            // if (!loginError.classList.contains("hidden"))
                loginError.classList.remove("hidden");
        }
    });
}


async function CheckForToken(): Promise<void> {
    try {
        const token = localStorage.getItem('token');
        if (!token)
            return ;
        const response = await fetch('http://localhost:3000/user-management/check-token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ token }),
        });
        const result = await response.json();
        console.log("CheckToken Result: ", result);
        if (result.valid) {
            if (result.username) {
                console.log("ResultUsername:", result.username);
                localStorage.setItem('username', result.username);

                const nameSpan = document.getElementById('username') as HTMLSpanElement;
                nameSpan.textContent = result.username;
                console.log("Welcome :", result.username);
                if (result.avatar)
                    localStorage.setItem('avatar', result.avatar);
                const AvatarSrc = document.getElementById('avatar') as HTMLImageElement;
                const Avatar = localStorage.getItem('avatar');
                if (Avatar)
                    AvatarSrc.src = Avatar;
                else
                    AvatarSrc.src = '../login.png';
            }
        }
        else
            localStorage.removeItem("token");
    }
    catch (error) {
        console.error("Bad token :", error);
    }
}


function validateRegister(result: { name: string; email: string; password: string}): void {
    const nameErrorMin = document.getElementById("nameErrorMin") as HTMLSpanElement;
    const emailError = document.getElementById("emailError") as HTMLSpanElement;
    const passwordError = document.getElementById("passwordError") as HTMLSpanElement;
    if (result.name)
        nameErrorMin.classList.remove("hidden");
    else {
        if (!nameErrorMin.classList.contains("hidden")) {
            nameErrorMin.classList.add("hidden");
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