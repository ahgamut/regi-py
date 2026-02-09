// websocket functions
function init_ws() {
    let g = Alpine.store('gamestate');
    let serverIP = window.location.host.toString();
    g.ws = new WebSocket(`ws://${serverIP}/ws/${g.userid}`);
    g.ws.onmessage = receive_ws;
}

function send_ws(message)
{
    let g = Alpine.store('gamestate');
    let raw = JSON.stringify(message);
    let res = g.ws.send(raw);
}

function receive_ws(event) {
    // twice because string inside string
    let info = JSON.parse(JSON.parse(event.data));
    let g = Alpine.store('gamestate');
    if (info.data !== null) {
        console.log(info);
    }
    switch(info.type) {
        case "loading":
            logMessage(`${info.remain} players still need to connect`, 'is-secondary');
            g.statusz = "LOADING";
            g.turnMessage = 'Press Ready and Wait for Other Players...';
            break;
        case "ready":
            g.statusz = "READY";
            break;
        case "log":
            // logMessage(JSON.stringify(info.data));
            processLog(info.data);
            g.history.push(info.data);
            break;
        case "select-attack":
            logMessage(`you have to attack`, 'is-primary');
            selectAttack(info.data);
            break;
        case "select-defend":
            logMessage(`you have to defend`, 'is-primary');
            selectDefend(info.data);
            break;
        default:
            logMessage("fug", 'is-danger');
            logMessage(JSON.stringify(info));
            break;
    }
}

//
function reset_game() {
    // console.log("Resetting game");
    let g = Alpine.store('gamestate');
    document.getElementById('messages').replaceChildren();
    let message = {userid: g.userid, type:'player-reset', choice:0};
    send_ws(message);
}

function download_json() {
    let g = Alpine.store('gamestate');
    let text = JSON.stringify(g.history);
    let filename = `regi-${Date.now().toString()}.json`;
    let element = document.createElement('a');
    element.setAttribute('href', 'data:application/json;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);
    element.click();
}

function request_start() {
    let g = Alpine.store('gamestate');
    send_ws({userid: g.userid, type: "player-join", choice:0});
}
function player_ready() {
    let g = Alpine.store('gamestate');
    send_ws({userid: g.userid, type: "player-ready", choice:0});
}




function submit_option() {
    let g = Alpine.store("gamestate");
    let cur_player = g.data;
    // get selected cards
    let cards = document.getElementById('player-cards').firstChild;
    let pickset = new Set([]);
    for (let card of cards.children) {
        if (card.classList.contains("is-focused")) {
            pickset.add(card.innerHTML);
        }
    }

    let picksOK = false;
    // console.log(cur_player);
    let comboLen = cur_player.combos.length;
    let pickIndex = -1;
    for (let i = 0; i < comboLen; i++) {
        let combset = new Set(cur_player.combos[i].map(x => x.value));
        if (pickset.isSubsetOf(combset) && combset.isSubsetOf(pickset)) {
            picksOK = true;
            pickIndex = i;
            break;
        }
    }

    if (!picksOK) {
        logMessage(`${Array.from(pickset)} is not a valid move!`, 'is-warning');
        updateCards(g.data.player);
        g.turnMessage = "Invalid Move! Pick Cards Again";
    } else {
        // send the option picked
        let msg = { userid: g.userid, type: "player-move", choice: pickIndex };
        send_ws(msg);
        logMessage(`You selected ${Array.from(pickset)}`);
        // after sending your option, disable the buttons
        let submitter = document.getElementById('main-button');
        let yielder = document.getElementById('side-button');
        setButtonActivity(submitter, false);
        setButtonActivity(yielder, false);
    }
}

function yield_option() {
    let g = Alpine.store("gamestate");
    let cur_player = g.data;
    
    let picksOK = false;
    let comboLen = cur_player.combos.length;
    let pickIndex = -1;
    for (let i = 0; i < comboLen; i++) {
        if (combos[i].length == 0) {
            picksOK = true;
            pickIndex = i;
            break;
        }
    }
    
    if (!picksOK) {
        logMessage("yield is not a valid move!", 'is-warning');
        g.turnMessage = "Invalid Yield! Pick Cards";
    } else {
        // send the option picked
        let msg = { userid: g.userid, choice: pickIndex };
        send_ws(msg);
        logMessage(`You selected ${Array.from(pickset)}`);
        // after sending your option, disable the buttons
        let submitter = document.getElementById('main-button');
        let yielder = document.getElementById('side-button');
        setButtonActivity(submitter, false);
        setButtonActivity(yielder, false);
    }
}

function setButtonActivity(button, active) {
    if (active) {
        button.classList.remove("is-loading")
    } else {
        button.classList.add("is-loading")
    }
}

function makeCurrentEnemy(ce) {
    let res = document.createElement("div");
    let hdr = document.createElement("h1");
    hdr.className = "title";
    hdr.innerHTML = `Current Enemy: ${ce.value} (${ce.hp} HP)`;
    res.appendChild(hdr);
    return res;
}

function makeUsedCombos(combos) {
    let res = document.createElement("div");

    let hdr = document.createElement("h1");
    hdr.className = "title";
    hdr.innerHTML = "Combos Used: ";
    res.appendChild(hdr);

    let els = document.createElement("ul");
    for (let combo of combos) {
        let el = document.createElement("li");
        for (let card of combo) {
            let b = document.createElement("div");
            b.className = "button is-link";
            b.innerHTML = card.value;
            el.appendChild(b);
        }
        els.appendChild(el);
    }
    res.append(els);
    return res;
}

function makeContextInfo(game) {
    let res = document.createElement("div");
    let els = document.createElement("ul");
    els.className = "subtitle";
    //
    let el1 = document.createElement("li");
    el1.appendChild(document.createTextNode(`Deck: ${game.draw_pile_size} cards`))
    //
    let el2 = document.createElement("li");
    el2.appendChild(document.createTextNode(`Discard Pile: ${game.discard_pile_size} cards`))
    //
    let el3 = document.createElement("li");
    el3.appendChild(document.createTextNode(`${game.enemy_pile_size} enemies left`))
    //
    els.appendChild(el1);
    els.appendChild(el2);
    els.appendChild(el3);
    res.append(els);
    // console.log(game);
    return res;
}

function makeOtherPlayerInfo(game) {
    let res = document.createElement("div");
    let els = document.createElement("ul");
    els.className = "subtitle";
    for (let player of game.players) {
        let el = document.createElement("li");
        el.appendChild(document.createTextNode(`Player ${player.id}: ${player.num_cards} cards`));
        els.appendChild(el);
    }
    res.appendChild(els);
    return res;
}

function updateBoard(game) {
    let g = Alpine.store('gamestate');
    let enemy_view = document.getElementById('enemy-view');
    let game_view = document.getElementById('game-view');
    // console.log(game);
    game_view.replaceChildren();
    game_view.appendChild(makeContextInfo(game));
    game_view.appendChild(makeOtherPlayerInfo(game));
    enemy_view.replaceChildren();
    enemy_view.appendChild(makeCurrentEnemy(game.current_enemy));
    enemy_view.appendChild(makeUsedCombos(game.used_combos));
}

function getCardButton(card) {
    let b = document.createElement("div");
    b.className = "button is-link";
    b.addEventListener("click", () => { 
        if (b.classList.contains("is-focused")) {
            b.classList.remove("is-focused") 
            b.classList.remove("is-dark")
            b.classList.add("is-link")
        } else {
            b.classList.add("is-focused")
            b.classList.add("is-dark")
            b.classList.remove("is-link")
        }
    });
    b.innerHTML = card;
    return b;
}

function updateCards(player) {
    let g = Alpine.store('gamestate');
    let target = document.getElementById('player-cards');
    target.replaceChildren();

    let bgroup = document.createElement("div");
    let cards = player.cards;
    bgroup.className = "buttons has-addons are-medium";
    for (let i = 0, len = cards.length; i < len; i++) {
        bgroup.appendChild(getCardButton(cards[i]));
    }
    target.appendChild(bgroup);
    // console.log(player);
}

function selectAttack(data) {
    let g = Alpine.store('gamestate');
    g.data = data;
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, data.yield_allowed);
    //
    g.playerShould = "You have to ATTACK";
    g.turnMessage = "Select cards for your attack.";
    // console.log(data);
    updateBoard(data.game);
    updateCards(data.player);
}
function selectDefend(data) {
    let g = Alpine.store('gamestate');
    g.data = data;
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, false);
    //
    g.playerShould = `You have to DEFEND ${data.damage} damage`;
    g.turnMessage = "Select cards for your defense.";
    // console.log(data);
    updateBoard(data.game);
    updateCards(data.player);
}

// logging
function processLog(data) {
    let g = Alpine.store('gamestate');
    if (data.game != null) {
        // console.log("WTF");
        // console.log(data);
        updateBoard(data.game);
    }
    switch (data.event) {
        case 'STARTGAME':
            if (data.game != null && data.game.active_player_id !== null) {
                g.statusz = "RUNNING";
                logMessage("Game has started", 'is-primary');
            }
            break;
        case 'ATTACK':
            let combo1 = data.combo.map(x => x.value);
            logMessage(`Player ${data.player.id} attacked ${data.enemy.value} with ${combo1}`, 'is-info');
            logMessage(`Player ${data.player.id} dealt ${data.damage} damage!`, 'is-info');
            break;
        case 'DEFEND':
            let combo2 = data.combo.map(x => x.value);
            logMessage(`${data.enemy.value} attacked Player ${data.player.id} for ${data.damage} damage`, 'is-info');
            logMessage(`Player ${data.player.id} blocked with ${combo2}`, 'is-info');
            break;
        case 'ENEMYKILL':
            let exact = data.enemy.hp === 0 ? " exact " : "";
            logMessage(`${data.enemy.value} killed` + exact + "!", 'is-success'); 
            break;
        case 'REPLENISH':
            logMessage(`${data.n_cards} cards added back to the draw pile`, 'is-info');
            break;
        case 'DRAWONE':
            logMessage(`Player ${data.player.id} drew a card`);
            break;
        case 'DECKEMPTY':
            logMessage(`Player ${data.player.id} cannot draw`);
            break;
        case 'ENDGAME':
            endGameStatusUpdate();
            logMessage("Game has ended", 'is-primary');
            break;
        case 'POSTGAME':
            endGameStatusUpdate();
            logMessage("Game over!", 'is-primary');
            // logMessage("Postgame: " + JSON.stringify(data.game));
            break;
        case 'FAILBLOCK':
            logMessage(`${data.enemy.value} attacked Player ${data.player.id} for ${data.damage}`, 'is-danger');
            logMessage(`Player ${data.player.id} can block at most ${data.maxblock}!`, 'is-danger');
            break;
        case 'FULLBLOCK':
            logMessage(`${data.enemy.value} is blocked by Player ${data.player.id}`, 'is-info');
            break;
        case 'STATE':
            g.statusz = "RUNNING";
            if (data.game.active_player_id !== null) {
                if (data.game.active_player_id != g.playerid) {
                    let submitter = document.getElementById('main-button');
                    let yielder = document.getElementById('side-button');
                    setButtonActivity(submitter, false);
                    setButtonActivity(yielder, false);
                    logMessage(`Wait for Player ${data.game.active_player_id} to play..`);
                } else {
                    updateCards(data.game.active_player);
                }
            }
        case 'DEBUG':
            // can I send something here?
            break;
        // default:
            // console.log(data.event);
    }
}

function addNotification(content, subtype) {
    let tray = document.getElementById('notification-tray')
    let duration = 1500;
    //
    let res = document.createElement('div')
    res.className = 'notification is-light subtitle';
    res.classList.add(subtype);
    res.role = "alert";

    let delButton = document.createElement('button');
    delButton.classList.add('delete');
    delButton.addEventListener("click", () => {
        tray.removeChild(res);
    });

    res.appendChild(delButton);
    res.appendChild(document.createTextNode(content));

    tray.appendChild(res);
    let removeDelay = tray.children.length * 1000 + 2 * duration + 1400;
    setTimeout(() => { 
        res.style.animation = `fadeOut 2000ms ease-in 1 forwards`;
    }, removeDelay);
    setTimeout(() => { res.remove(); }, removeDelay + 1400);
}

function logMessage(content, subtype=null) {
    let g = Alpine.store('gamestate');
    // add to log at the bottom
    let messages = document.getElementById('messages')
    let message = document.createElement('li')
    message.appendChild(document.createTextNode(content))
    messages.appendChild(message)
    if (subtype != null) {
        // add to notification tray.
        setTimeout(() => {addNotification(content, subtype);}, 1000);
    }
    // show player message
    g.turnMessage = content;
}

// UI
function gameStatusShowable() {
    let g = Alpine.store('gamestate');
    let showStates = ["READY", "RUNNING", "ENDED"];
    return showStates.includes(g.statusz);
}

function endGameStatusUpdate() {
    let g = Alpine.store('gamestate');
    g.statusz = "ENDED";
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, true);
}

function mainButtonText() {
    let g = Alpine.store('gamestate');
    let result = "";
    switch (g.statusz) {
        case "LOADING":
            result = "Connect";
            break;
        case "READY":
            result = "Ready!"
            break;
        case "RUNNING":
            result = "Submit";
            break;
        case "ENDED":
        case "ERROR":
            result = "Restart";
            break;
        default:
            result = "???";
    }
    return result;
}
function mainButtonRedirect() {
    let button = document.getElementById("main-button");
    if (button.classList.contains("is-loading")) {
        return;
    }
    let g = Alpine.store('gamestate');
    switch (g.statusz) {
        case "LOADING":
            request_start();
            break;
        case "READY":
            player_ready();
            break;
        case "RUNNING":
            submit_option();
            break;
        case "ENDED":
        case "ERROR":
            reset_game();
            break;
    }
}

function sideButtonText() {
    let g = Alpine.store('gamestate');
    let result = "";
    switch (g.statusz) {
        case "LOADING":
        case "READY":
        case "ERROR":
            result = "<hide>";
            break;
        case "RUNNING":
            result = "Yield";
            break;
        case "ENDED":
            result = "Download JSON";
            break;
        default:
            result = "???";
    }
    return result;
}

function sideButtonRedirect() {
    let g = Alpine.store('gamestate');
    let button = document.getElementById("side-button");
    if (button.classList.contains("is-loading")) {
        return;
    }
    switch (g.statusz) {
        case "LOADING":
            break;
        case "READY":
            break;
        case "RUNNING":
            yield_option();
            break;
        case "ENDED":
        case "ERROR":
            download_json();
            break;
    }
}


