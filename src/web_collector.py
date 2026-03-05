from __future__ import annotations

import time
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

from playwright.sync_api import Page, BrowserContext, sync_playwright, TimeoutError as PlaywrightTimeoutError

from .models import Message, Conversation

# --- Configuration Class ---

@dataclass
class WebCollectorConfig:
    login_url: str
    headless: bool
    max_conversations: int
    max_pages: int
    selectors: Dict[str, str]
    filters: List[Dict[str, str]]
    wait_after_filter_seconds: float = 1.0
    post_login_url_contains: Optional[str] = None
    timezone: str = "Asia/Kuala_Lumpur"
    filter_ready_selector: Optional[str] = None
    login_timeout_seconds: int = 300
    storage_state_path: Optional[str] = ".auth/storage_state.json"


def _debug(section: str, message: str) -> None:
    print(f"[{section}] {message}")

# --- Helper Utilities ---

def _safe_fill(page: Page, selector: str, value: str) -> None:
    if not selector:
        return
    loc_selector = f"xpath={selector}" if selector.startswith("/") or selector.startswith("(") else selector
    loc = page.locator(loc_selector).first
    try:
        loc.wait_for(state="visible", timeout=5000)
        loc.click()
        loc.fill(value)
        loc.press("Enter")
    except Exception:
        loc.evaluate(
            """(el, val) => {
                el.removeAttribute('readonly');
                el.value = val;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }""",
            value,
        )
        loc.focus()
        page.keyboard.press("Enter")

def _safe_click(page: Page, selector: str, timeout: int = 10000) -> bool:
    if not selector:
        return False
    loc_selector = f"xpath={selector}" if selector.startswith("/") or selector.startswith("(") else selector
    loc = page.locator(loc_selector).first
    try:
        loc.wait_for(state="visible", timeout=timeout)
        loc.scroll_into_view_if_needed()
        loc.click()
        return True
    except Exception:
        try:
            loc.wait_for(state="attached", timeout=3000)
            loc.click(force=True)
            return True
        except Exception as e:
            _debug("CLICK_ERR", f"Could not click {selector}: {e}")
            return False


def _screenshot(page: Page, name: str) -> None:
    try:
        path = Path("data") / f"debug_{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path), full_page=False)
        _debug("SCREENSHOT", f"Saved {path}")
    except Exception as e:
        _debug("SCREENSHOT", f"Failed: {e}")


def _dump_dropdown_options(page: Page) -> List[str]:
    """Log every visible option inside any open ant-select-dropdown."""
    try:
        items = page.locator(
            "xpath=//div[contains(@class, 'ant-select-dropdown') "
            "and not(contains(@class, 'ant-select-dropdown-hidden'))]"
            "//div[contains(@class, 'ant-select-item')]"
        )
        texts = []
        for i in range(items.count()):
            t = items.nth(i).inner_text().strip()
            if t:
                texts.append(t)
        _debug("SELECT", f"Dropdown options visible ({len(texts)}): {texts}")
        return texts
    except Exception as e:
        _debug("SELECT", f"Could not dump options: {e}")
        return []


def _close_dropdown(page: Page) -> None:
    """Close any open Ant Design Select dropdown (needed for multi-select)."""
    page.keyboard.press("Escape")
    time.sleep(0.3)


def _open_ant_dropdown(page: Page, trigger_loc: str, label: str) -> bool:
    """Try multiple strategies to open an Ant Design Select dropdown.

    Returns True if the dropdown appeared.
    """
    dropdown_css = "div.ant-select-dropdown:not(.ant-select-dropdown-hidden)"
    trigger = page.locator(trigger_loc).first

    def _dropdown_visible() -> bool:
        try:
            page.wait_for_selector(dropdown_css, state="visible", timeout=3000)
            return True
        except PlaywrightTimeoutError:
            return False

    # Log what element we actually found
    try:
        tag = trigger.evaluate("el => el.tagName")
        cls = trigger.evaluate("el => el.className")
        _debug("SELECT", f"Trigger element: <{tag}> class='{cls}'")
    except Exception:
        pass

    trigger.wait_for(state="attached", timeout=10000)
    try:
        trigger.scroll_into_view_if_needed()
    except Exception:
        pass

    # Attempt A: if trigger IS an <input>, focus then click it
    _debug("SELECT", f"[{label}] Attempt A: focus + click trigger")
    try:
        trigger.focus()
        time.sleep(0.2)
        trigger.click(force=True)
        time.sleep(0.5)
        if _dropdown_visible():
            return True
    except Exception:
        pass

    # Attempt B: click the <input> inside the trigger (multi-select search input)
    _debug("SELECT", f"[{label}] Attempt B: click input inside trigger")
    try:
        search_input = trigger.locator("input").first
        search_input.focus()
        time.sleep(0.2)
        search_input.click(force=True)
        time.sleep(0.5)
        if _dropdown_visible():
            return True
    except Exception:
        pass

    # Attempt C: click the parent .ant-select wrapper
    _debug("SELECT", f"[{label}] Attempt C: click parent .ant-select wrapper")
    try:
        parent_select = trigger.evaluate(
            "el => el.closest('.ant-select') || el.parentElement.closest('.ant-select')"
        )
        if parent_select:
            wrapper = page.locator(
                f"xpath={trigger_loc}/ancestor::div[contains(@class,'ant-select')]"
                if trigger_loc.startswith("xpath=") else
                f"{trigger_loc} >> xpath=ancestor::div[contains(@class,'ant-select')]"
            ).first
            wrapper.click()
            time.sleep(0.5)
            if _dropdown_visible():
                return True
    except Exception:
        pass

    # Attempt D: JS mousedown on .ant-select (React/Ant event trigger)
    _debug("SELECT", f"[{label}] Attempt D: JS mousedown on .ant-select")
    try:
        trigger.evaluate(
            """el => {
                const sel = el.closest('.ant-select') || el;
                sel.querySelector('.ant-select-selector')
                    .dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
            }"""
        )
        time.sleep(0.5)
        if _dropdown_visible():
            return True
    except Exception:
        pass

    # Attempt E: JS focus on search input via DOM query
    _debug("SELECT", f"[{label}] Attempt E: JS focus search input")
    try:
        trigger.evaluate(
            """el => {
                const sel = el.closest('.ant-select') || el;
                const input = sel.querySelector('input.ant-select-selection-search-input') || sel.querySelector('input');
                if (input) { input.focus(); input.click(); }
            }"""
        )
        time.sleep(0.5)
        if _dropdown_visible():
            return True
    except Exception:
        pass

    return False


def _ant_select_only(page: Page, label_text: str, desired_value: str, timeout: int = 10000) -> None:
    """For an Ant Design multi-select identified by its form label, ensure only
    ``desired_value`` is selected.  Uses pure JS to find the element and remove
    unwanted tags, avoiding Playwright locator chain issues.
    """
    _screenshot(page, "select_only_before")

    # --- Step 1: find the select element near the label using JS ---
    find_result = page.evaluate(
        """(labelText) => {
            const candidates = document.querySelectorAll('label, span, div, td');
            for (const el of candidates) {
                const ownText = Array.from(el.childNodes)
                    .filter(n => n.nodeType === 3)
                    .map(n => n.textContent.trim())
                    .join('');
                if (!ownText.includes(labelText)) continue;
                if (el.closest('table') || el.closest('thead') || el.closest('th')) continue;

                let parent = el.parentElement;
                for (let i = 0; i < 10; i++) {
                    if (!parent) break;
                    const sel = parent.querySelector('.ant-select');
                    if (sel && sel.querySelector('.ant-select-selector')) {
                        const items = sel.querySelectorAll('.ant-select-selection-item');
                        return {
                            found: true,
                            classes: sel.className,
                            tags: Array.from(items).map(t => ({
                                title: t.getAttribute('title') || t.textContent.trim(),
                                hasRemove: !!t.querySelector('.ant-select-selection-item-remove')
                            }))
                        };
                    }
                    parent = parent.parentElement;
                }
            }
            return { found: false };
        }""",
        label_text,
    )
    _debug("SELECT_ONLY", f"Find result: {find_result}")

    if not find_result.get("found"):
        raise RuntimeError(f"Could not find an Ant Design Select near label '{label_text}'")

    # --- Step 2: remove every tag except the desired value ---
    max_rounds = 15
    for round_num in range(max_rounds):
        removed = page.evaluate(
            """({label, keep}) => {
                const candidates = document.querySelectorAll('label, span, div, td');
                for (const el of candidates) {
                    const ownText = Array.from(el.childNodes)
                        .filter(n => n.nodeType === 3)
                        .map(n => n.textContent.trim())
                        .join('');
                    if (!ownText.includes(label)) continue;
                    if (el.closest('table') || el.closest('thead') || el.closest('th')) continue;

                    let parent = el.parentElement;
                    for (let i = 0; i < 10; i++) {
                        if (!parent) break;
                        const sel = parent.querySelector('.ant-select');
                        if (sel && sel.querySelector('.ant-select-selector')) {
                            const items = sel.querySelectorAll('.ant-select-selection-item');
                            for (const item of items) {
                                const title = (item.getAttribute('title') || item.textContent || '').trim();
                                if (title.toLowerCase() === keep.toLowerCase()) continue;
                                const btn = item.querySelector('.ant-select-selection-item-remove');
                                if (btn) {
                                    btn.click();
                                    return { removed: title };
                                }
                            }
                            return { removed: null, remaining: Array.from(items).map(t => t.getAttribute('title') || t.textContent.trim()) };
                        }
                        parent = parent.parentElement;
                    }
                }
                return { removed: null, error: 'select not found' };
            }""",
            {"label": label_text, "keep": desired_value},
        )
        _debug("SELECT_ONLY", f"Round {round_num + 1}: {removed}")

        if removed.get("removed"):
            time.sleep(0.5)
        else:
            break

    _screenshot(page, "select_only_after_remove")

    # --- Step 3: check current state ---
    state = page.evaluate(
        """(label) => {
            const candidates = document.querySelectorAll('label, span, div, td');
            for (const el of candidates) {
                const ownText = Array.from(el.childNodes)
                    .filter(n => n.nodeType === 3)
                    .map(n => n.textContent.trim())
                    .join('');
                if (!ownText.includes(label)) continue;
                if (el.closest('table') || el.closest('thead') || el.closest('th')) continue;

                let parent = el.parentElement;
                for (let i = 0; i < 10; i++) {
                    if (!parent) break;
                    const sel = parent.querySelector('.ant-select');
                    if (sel && sel.querySelector('.ant-select-selector')) {
                        const items = sel.querySelectorAll('.ant-select-selection-item');
                        return {
                            tags: Array.from(items).map(t => t.getAttribute('title') || t.textContent.trim())
                        };
                    }
                    parent = parent.parentElement;
                }
            }
            return { tags: [] };
        }""",
        label_text,
    )
    _debug("SELECT_ONLY", f"After removal, tags: {state.get('tags')}")

    if desired_value in (state.get("tags") or []):
        _debug("SELECT_ONLY", f"'{desired_value}' is selected. Done.")
        return

    # --- Step 4: desired value not present, add via keyboard ---
    _debug("SELECT_ONLY", f"'{desired_value}' missing — trying keyboard add")

    # Focus the search input inside the select via JS
    page.evaluate(
        """(label) => {
            const candidates = document.querySelectorAll('label, span, div, td');
            for (const el of candidates) {
                const ownText = Array.from(el.childNodes)
                    .filter(n => n.nodeType === 3)
                    .map(n => n.textContent.trim())
                    .join('');
                if (!ownText.includes(label)) continue;
                if (el.closest('table') || el.closest('thead') || el.closest('th')) continue;

                let parent = el.parentElement;
                for (let i = 0; i < 10; i++) {
                    if (!parent) break;
                    const sel = parent.querySelector('.ant-select');
                    if (sel) {
                        const input = sel.querySelector('input');
                        if (input) { input.focus(); return true; }
                    }
                    parent = parent.parentElement;
                }
            }
            return false;
        }""",
        label_text,
    )
    time.sleep(0.3)
    page.keyboard.type(desired_value, delay=80)
    time.sleep(1.0)
    page.keyboard.press("Enter")
    time.sleep(0.5)
    page.keyboard.press("Escape")
    time.sleep(0.3)

    _screenshot(page, "select_only_after_add")
    _debug("SELECT_ONLY", f"Keyboard add attempted for '{desired_value}'")


def _verify_selected(page: Page, trigger_loc: str, option_text: str) -> bool:
    """Check whether a tag with the option text appeared in the select."""
    try:
        tag_xpath = (
            f"xpath={trigger_loc}/ancestor::div[contains(@class,'ant-select')]"
            f"//span[contains(@class,'ant-select-selection-item')][contains(@title,'{option_text}')]"
        ) if trigger_loc.startswith("xpath=") else (
            f"{trigger_loc} >> xpath=ancestor::div[contains(@class,'ant-select')]"
            f"//span[contains(@class,'ant-select-selection-item')][contains(@title,'{option_text}')]"
        )
        count = page.locator(tag_xpath).count()
        if count > 0:
            _debug("SELECT", f"Verified: '{option_text}' tag found in selector")
            return True
    except Exception:
        pass

    try:
        title_sel = f"span.ant-select-selection-item[title='{option_text}']"
        count = page.locator(title_sel).count()
        if count > 0:
            _debug("SELECT", f"Verified (css): '{option_text}' tag found")
            return True
    except Exception:
        pass

    return False


def _ant_select(page: Page, trigger_selector: str, option_text: str, timeout: int = 10000) -> None:
    """Open an Ant Design Select dropdown and pick an option by visible text.

    Tries dropdown-based selection first, then falls back to keyboard-only.
    """
    trigger_loc = (
        f"xpath={trigger_selector}"
        if trigger_selector.startswith("/") or trigger_selector.startswith("(")
        else trigger_selector
    )

    # Close any stale popover / dropdown first
    page.keyboard.press("Escape")
    time.sleep(0.3)

    _screenshot(page, "select_before_open")
    dropdown_appeared = _open_ant_dropdown(page, trigger_loc, option_text)

    dropdown_css = "div.ant-select-dropdown:not(.ant-select-dropdown-hidden)"

    if dropdown_appeared:
        time.sleep(0.3)
        available = _dump_dropdown_options(page)
        _screenshot(page, "select_dropdown_open")

        # --- Click-based strategies (only if dropdown is visible) ---

        # Strategy 1: XPath match
        option_xpaths = [
            f"//div[contains(@class,'ant-select-dropdown') and not(contains(@class,'ant-select-dropdown-hidden'))]"
            f"//div[contains(@class,'ant-select-item-option-content')][normalize-space()='{option_text}']",
            f"//div[contains(@class,'ant-select-dropdown') and not(contains(@class,'ant-select-dropdown-hidden'))]"
            f"//div[contains(@class,'ant-select-item')][normalize-space()='{option_text}']",
        ]
        for xp in option_xpaths:
            loc = page.locator(f"xpath={xp}").first
            try:
                loc.wait_for(state="visible", timeout=3000)
                loc.scroll_into_view_if_needed()
                loc.click()
                _debug("SELECT", f"Strategy 1 (xpath) selected '{option_text}'")
                time.sleep(0.3)
                _close_dropdown(page)
                return
            except Exception:
                continue

        # Strategy 2: Playwright text selector scoped to dropdown
        try:
            dropdown = page.locator(dropdown_css).last
            option_by_text = dropdown.get_by_text(option_text, exact=True).first
            option_by_text.wait_for(state="visible", timeout=3000)
            option_by_text.scroll_into_view_if_needed()
            option_by_text.click()
            _debug("SELECT", f"Strategy 2 (get_by_text) selected '{option_text}'")
            time.sleep(0.3)
            _close_dropdown(page)
            return
        except Exception:
            pass

    # --- Keyboard-only fallback (works even if dropdown isn't detected) ---
    _debug("SELECT", "Trying keyboard-only selection (type + Enter)")

    trigger = page.locator(trigger_loc).first

    # Make sure the search input is focused
    try:
        trigger.focus()
    except Exception:
        pass
    try:
        tag = trigger.evaluate("el => el.tagName")
        if tag.upper() != "INPUT":
            inp = trigger.evaluate(
                """el => {
                    const s = el.closest('.ant-select') || el;
                    const i = s.querySelector('input');
                    if (i) { i.focus(); return true; }
                    return false;
                }"""
            )
    except Exception:
        pass

    time.sleep(0.3)

    # Type the option text to search/filter
    page.keyboard.type(option_text, delay=80)
    time.sleep(1.0)
    _screenshot(page, "select_after_keyboard")
    _dump_dropdown_options(page)

    # Press Enter to select the first matching option
    page.keyboard.press("Enter")
    time.sleep(0.5)

    # Close the dropdown
    _close_dropdown(page)
    time.sleep(0.3)

    _screenshot(page, "select_after_enter")

    # Verify selection
    if _verify_selected(page, trigger_loc, option_text):
        _debug("SELECT", f"Keyboard selection of '{option_text}' succeeded")
        return

    # Even if verification fails, the selection might still have worked
    # (verification depends on DOM structure matching our expectations)
    _debug("SELECT", f"Could not verify '{option_text}' tag, but continuing anyway")
    _screenshot(page, "select_unverified")

def _resolve_value(raw_value: str, timezone: str) -> str:
    value = str(raw_value or "")
    now = datetime.now(ZoneInfo(timezone))
    m_today = re.fullmatch(r"\{\{TODAY:(.+)\}\}", value)
    if m_today:
        return now.strftime(m_today.group(1))
    m_minus = re.fullmatch(r"\{\{TODAY_MIN_DAYS:(\d+):(.+)\}\}", value)
    if m_minus:
        dt = now - timedelta(days=int(m_minus.group(1)))
        return dt.strftime(m_minus.group(2))
    return value

def _apply_actions(page: Page, actions: List[Dict[str, str]]) -> None:
    for idx, action in enumerate(actions, start=1):
        action_type = (action.get("action") or "").lower()
        selector = action.get("selector") or ""
        value = action.get("value") or ""
        optional = bool(action.get("optional", False))
        timezone = action.get("timezone") or "Asia/Kuala_Lumpur"
        resolved_value = _resolve_value(value, timezone)

        try:
            _debug("FILTER", f"Action {idx}: {action_type} | selector={selector}")
            if action_type == "fill":
                _safe_fill(page, selector, resolved_value)
            elif action_type == "click":
                _safe_click(page, selector)
            elif action_type == "select":
                _ant_select(page, selector, resolved_value)
            elif action_type == "select_only":
                _ant_select_only(page, selector, resolved_value)
            elif action_type == "press":
                loc_sel = f"xpath={selector}" if selector.startswith("/") or selector.startswith("(") else selector
                target = page.locator(loc_sel).first if selector else page.keyboard
                target.press(resolved_value)
            elif action_type == "wait":
                time.sleep(float(resolved_value))
            else:
                _debug("FILTER", f"Action {idx}: unknown action '{action_type}', skipping")
            time.sleep(0.5)
        except Exception as e:
            if optional:
                _debug("FILTER", f"Action {idx} optional failure (skipping): {e}")
                continue
            raise RuntimeError(f"Filter action failed at step {idx}: {e}")

# --- Core Scraper Logic ---

def _read_conversation(page: Page, selectors: Dict[str, str], fallback_id: str) -> Optional[Conversation]:
    msg_selector = selectors.get("message_row")
    if not msg_selector:
        return None

    loc_prefix = "xpath=" if msg_selector.startswith("/") or msg_selector.startswith("(") else ""

    try:
        page.wait_for_selector(f"{loc_prefix}{msg_selector}", timeout=8000)
    except Exception:
        _debug("READ", "No messages found with configured selector, dumping drawer DOM classes...")
        try:
            classes = page.evaluate("""() => {
                const drawer = document.querySelector('.ant-drawer-open .ant-drawer-body')
                             || document.querySelector('.ant-drawer-body')
                             || document.querySelector('.ant-modal-body');
                if (!drawer) return 'No drawer/modal body found';
                const divs = drawer.querySelectorAll('div[class]');
                const classSet = new Set();
                divs.forEach(d => d.className.split(' ').forEach(c => { if (c) classSet.add(c); }));
                return Array.from(classSet).sort().join(', ');
            }""")
            _debug("READ", f"Drawer CSS classes: {classes}")
        except Exception:
            pass
        return None

    conv_id = fallback_id
    id_sel = selectors.get("conversation_id")
    if id_sel:
        try:
            id_loc_prefix = "xpath=" if id_sel.startswith("/") or id_sel.startswith("(") else ""
            conv_id = page.locator(f"{id_loc_prefix}{id_sel}").first.inner_text().strip() or fallback_id
        except Exception:
            pass

    rows = page.locator(f"{loc_prefix}{msg_selector}")
    row_count = rows.count()
    _debug("READ", f"Matched {row_count} message row(s) with selector: {msg_selector}")

    # On first conversation, dump the first row's HTML for debugging
    if row_count > 0:
        try:
            sample_html = rows.first.evaluate("el => el.outerHTML.substring(0, 500)")
            _debug("READ", f"First row HTML: {sample_html}")
        except Exception:
            pass

    messages: List[Message] = []
    for i in range(row_count):
        row = rows.nth(i)
        try:
            text_sel = selectors.get("message_text", "div")
            text = row.locator(text_sel).first.inner_text().strip()
            if not text:
                continue

            sender = "unknown"
            try:
                sender = row.locator(selectors.get("message_sender", "span")).first.inner_text().strip() or "unknown"
            except Exception:
                pass

            timestamp = datetime.now().isoformat()
            try:
                timestamp = row.locator(selectors.get("message_timestamp", "span")).first.inner_text().strip() or timestamp
            except Exception:
                pass

            messages.append(Message(
                conversation_id=conv_id,
                sender=sender,
                timestamp=timestamp,
                text=text,
            ))
        except Exception as e:
            if i == 0:
                _debug("READ", f"Row {i} text extraction failed: {e}")
            continue

    _debug("READ", f"Extracted {len(messages)} message(s) for conversation {conv_id}")
    return Conversation(conversation_id=conv_id, messages=messages) if messages else None

def _close_drawer(page: Page, selectors: Dict[str, str]) -> None:
    """Close any open Ant Design Drawer, with multiple fallbacks."""
    close_sel = selectors.get("transcript_close_button")

    # Strategy 1: configured close button selector
    if close_sel:
        if _safe_click(page, close_sel):
            time.sleep(0.5)
            return

    # Strategy 2: any visible close button inside an open drawer
    fallbacks = [
        "//div[contains(@class,'ant-drawer-open')]//button[contains(@class,'ant-drawer-close')]",
        "//div[contains(@class,'ant-drawer-open')]//button[@aria-label='Close']",
        "//div[contains(@class,'ant-drawer-open')]//span[contains(@class,'anticon-close')]",
    ]
    for fb in fallbacks:
        if _safe_click(page, fb, timeout=2000):
            time.sleep(0.5)
            return

    # Strategy 3: press Escape
    _debug("DRAWER", "Fallback: pressing Escape to close drawer")
    page.keyboard.press("Escape")
    time.sleep(0.5)

    # Strategy 4: JS click on .ant-drawer-close inside open drawer
    try:
        page.evaluate("""() => {
            const btn = document.querySelector('.ant-drawer-open .ant-drawer-close')
                     || document.querySelector('.ant-drawer-open button[aria-label="Close"]');
            if (btn) btn.click();
        }""")
        time.sleep(0.5)
    except Exception:
        pass


def _open_transcript_by_xpath(context: BrowserContext, page: Page, selectors: Dict[str, str], index: int, fallback_id: str) -> Optional[Conversation]:
    base_xpath = selectors.get("transcript_links")
    if not base_xpath:
        return None

    indexed_xpath = f"({base_xpath})[{index + 1}]"
    btn = page.locator(f"xpath={indexed_xpath}")

    try:
        if btn.count() == 0:
            return None

        new_page = None
        try:
            with context.expect_page(timeout=3000) as popup_info:
                btn.click(force=True)
            new_page = popup_info.value
        except PlaywrightTimeoutError:
            pass

        if new_page:
            new_page.wait_for_load_state("networkidle")
            conv = _read_conversation(new_page, selectors, fallback_id)
            new_page.close()
            return conv

        # Same-page drawer/modal — wait for it to appear
        try:
            page.wait_for_selector(
                "div.ant-drawer-open, div.ant-modal-wrap:not([style*='display: none'])",
                timeout=5000,
            )
        except PlaywrightTimeoutError:
            _debug("ROW", f"No drawer/modal appeared for row {index + 1}")
            return None

        # Take a screenshot of the first drawer for debugging selectors
        if index == 0:
            _screenshot(page, "drawer_content")

        time.sleep(1.0)
        conv = _read_conversation(page, selectors, fallback_id)

        # Always close the drawer before returning
        _close_drawer(page, selectors)
        return conv

    except Exception as e:
        _debug("ROW", f"Error processing row {index + 1}: {e}")
        # Ensure drawer is closed even on error
        try:
            _close_drawer(page, selectors)
        except Exception:
            pass
        return None

def collect_conversations_from_web(cfg: WebCollectorConfig, username: Optional[str] = None, password: Optional[str] = None) -> List[Conversation]:
    conversations = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=cfg.headless)
        storage_path = Path(cfg.storage_state_path) if cfg.storage_state_path else None
        has_saved_state = bool(storage_path and storage_path.exists())

        context_kwargs: Dict[str, Any] = {"viewport": {"width": 1280, "height": 800}}
        if has_saved_state:
            context_kwargs["storage_state"] = str(storage_path)
            _debug("LOGIN", f"Using saved login session from {storage_path}")

        context = browser.new_context(**context_kwargs)
        page = context.new_page()
        
        _debug("NAV", f"Navigating to {cfg.login_url}")
        page.goto(cfg.login_url, wait_until="networkidle")

        # Session Validation
        login_completed = False
        if has_saved_state:
            try:
                _debug("LOGIN", "Checking whether saved session is still valid")
                if cfg.post_login_url_contains:
                    page.wait_for_url(f"**{cfg.post_login_url_contains}**", timeout=10000)
                if cfg.filter_ready_selector:
                    page.wait_for_selector(cfg.filter_ready_selector, timeout=10000)
                login_completed = True
                _debug("LOGIN", "Saved session valid; skipping manual login")
            except PlaywrightTimeoutError:
                _debug("LOGIN", "Saved session expired; manual login required")

        if not login_completed:
            _debug("LOGIN", f"Waiting up to {cfg.login_timeout_seconds}s for manual login...")
            if cfg.post_login_url_contains:
                try:
                    page.wait_for_url(f"**{cfg.post_login_url_contains}**", timeout=cfg.login_timeout_seconds * 1000)
                except PlaywrightTimeoutError: pass
            if cfg.filter_ready_selector:
                page.wait_for_selector(cfg.filter_ready_selector, timeout=cfg.login_timeout_seconds * 1000)

            if storage_path:
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                context.storage_state(path=str(storage_path))
                _debug("LOGIN", f"Saved login session to {storage_path}")

        # Apply Filters
        if cfg.filters:
            _debug("FILTER", f"Applying {len(cfg.filters)} filter actions")
            _apply_actions(page, cfg.filters)
            time.sleep(cfg.wait_after_filter_seconds)

        # Scrape Pages
        for p_idx in range(cfg.max_pages):
            xpath_selector = cfg.selectors.get("transcript_links")
            if not xpath_selector: break

            _debug("PAGE", f"Loading page {p_idx + 1}")
            try:
                page.wait_for_selector(f"xpath={xpath_selector}", timeout=15000)
            except PlaywrightTimeoutError:
                _debug("PAGE", f"No transcripts found on page {p_idx + 1}. Stopping.")
                break
            
            row_count = page.locator(f"xpath={xpath_selector}").count()
            _debug("PAGE", f"Page {p_idx + 1}: found {row_count} transcript rows")

            for i in range(row_count):
                if len(conversations) >= cfg.max_conversations:
                    break
                
                _debug("ROW", f"Reading row {i + 1}/{row_count} on page {p_idx + 1}")
                conv = _open_transcript_by_xpath(context, page, cfg.selectors, i, f"conv_{len(conversations)}")
                if conv:
                    conversations.append(conv)
                    _debug("ROW", f"Collected conversation {conv.conversation_id} with {len(conv.messages)} messages")

            if len(conversations) >= cfg.max_conversations:
                break
            
            # Pagination
            next_btn_sel = cfg.selectors.get("next_page")
            if next_btn_sel:
                next_btn = page.locator(next_btn_sel).first
                if next_btn.is_visible() and next_btn.is_enabled():
                    _debug("PAGINATION", "Clicking next page")
                    next_btn.click()
                    time.sleep(cfg.wait_after_filter_seconds)
                else: break
            else: break

        browser.close()
    
    return conversations